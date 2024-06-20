import torch
import copy
import random
import numpy as np


# Diffusion utils
# ------------------------------------------------------------------------
def encode_prompt(prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train=True):
    prompt_embeds_list = []

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def compute_embeddings(
    prompt_batch, original_sizes, crop_coords, proportion_empty_prompts, text_encoders, tokenizers, is_train=True,
    device='cuda'
):
    target_size = (1024, 1024)
    original_sizes = original_sizes #list(map(list, zip(*original_sizes)))
    crops_coords_top_left = crop_coords #list(map(list, zip(*crop_coords)))

    original_sizes = torch.tensor(original_sizes, dtype=torch.long)
    crops_coords_top_left = torch.tensor(crops_coords_top_left, dtype=torch.long)

    prompt_embeds, pooled_prompt_embeds = encode_prompt(
        prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train
    )
    add_text_embeds = pooled_prompt_embeds

    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    add_time_ids = list(target_size)
    add_time_ids = torch.tensor([add_time_ids])
    add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)
    add_time_ids = torch.cat([original_sizes, crops_coords_top_left, add_time_ids], dim=-1)
    add_time_ids = add_time_ids.to(device, dtype=prompt_embeds.dtype)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

    return {"prompt_embeds": prompt_embeds, **unet_added_cond_kwargs}

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def guidance_scale_embedding(w, embedding_dim=512, dtype=torch.float32):
    """
    See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

    Args:
        timesteps (`torch.Tensor`):
            generate embedding vectors at these timesteps
        embedding_dim (`int`, *optional*, defaults to 512):
            dimension of the embeddings to generate
        dtype:
            data type of the generated embeddings

    Returns:
        `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(w.shape) == 1
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb

def predicted_origin(model_output, timesteps, boundary_timesteps, sample, prediction_type, alphas, sigmas):
    sigmas_s = extract_into_tensor(sigmas, boundary_timesteps, sample.shape)
    alphas_s = extract_into_tensor(alphas, boundary_timesteps, sample.shape)

    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)

    # Set hard boundaries to ensure equivalence with forward (direct) CD
    alphas_s[boundary_timesteps == 0] = 1.0
    sigmas_s[boundary_timesteps == 0] = 0.0

    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas # x0 prediction
        pred_x_0 = alphas_s * pred_x_0 + sigmas_s * model_output # Euler step to the boundary step
    elif prediction_type == "v_prediction":
        assert boundary_timesteps == 0, "v_prediction does not support multiple endpoints at the moment"
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")

    return pred_x_0


class DDIMSolver:
    def __init__(
            self, alpha_cumprods, timesteps=1000, ddim_timesteps=50,
            num_endpoints=1, num_inverse_endpoints=1,
            max_inverse_timestep_index=49,
            endpoints=None, inverse_endpoints=None
    ):
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(
            np.int64) - 1  # [19, ..., 999]
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        self.ddim_alpha_cumprods_next = np.asarray(
            alpha_cumprods[self.ddim_timesteps[1:]].tolist() + [0.0]
        )
        # convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)
        self.ddim_alpha_cumprods_next = torch.from_numpy(self.ddim_alpha_cumprods_next)

        # Set endpoints for direct CTM
        if endpoints is None:
            timestep_interval = ddim_timesteps // num_endpoints + int(ddim_timesteps % num_endpoints > 0)
            endpoint_idxs = torch.arange(timestep_interval, ddim_timesteps, timestep_interval) - 1
            self.endpoints = torch.tensor([0] + self.ddim_timesteps[endpoint_idxs].tolist())
        else:
            self.endpoints = torch.tensor([int(endpoint) for endpoint in endpoints.split(',')])
            assert len(self.endpoints) == num_endpoints

        # Set endpoints for inverse CTM
        if inverse_endpoints is None:
            timestep_interval = ddim_timesteps // num_inverse_endpoints + int(
                ddim_timesteps % num_inverse_endpoints > 0)
            inverse_endpoint_idxs = torch.arange(timestep_interval, ddim_timesteps, timestep_interval) - 1
            inverse_endpoint_idxs = torch.tensor(inverse_endpoint_idxs.tolist() + [max_inverse_timestep_index])
            self.inverse_endpoints = self.ddim_timesteps[inverse_endpoint_idxs]
        else:
            self.inverse_endpoints = torch.tensor([int(endpoint) for endpoint in inverse_endpoints.split(',')])
            assert len(self.inverse_endpoints) == num_inverse_endpoints

    def to(self, device):
        self.endpoints = self.endpoints.to(device)
        self.inverse_endpoints = self.inverse_endpoints.to(device)

        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        self.ddim_alpha_cumprods_next = self.ddim_alpha_cumprods_next.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev

    def inverse_ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_next = extract_into_tensor(self.ddim_alpha_cumprods_next, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_next).sqrt() * pred_noise
        x_next = alpha_cumprod_next.sqrt() * pred_x0 + dir_xt
        return x_next
# ------------------------------------------------------------------------

# Distillation specific
# ------------------------------------------------------------------------
def inverse_sample_deterministic(
        pipe,
        images,
        prompt,
        generator=None,
        num_scales=50,
        num_inference_steps=1,
        timesteps=None,
        start_timestep=19,
        max_inverse_timestep_index=49,
        return_start_latent=False,
        guidance_scale=None,  # Used only if the student has w_embedding
        compute_embeddings_fn=None,
        is_sdxl=False,
        inverse_endpoints=None,
        seed=0,
):
    # assert isinstance(pipe, StableDiffusionImg2ImgPipeline), f"Does not support the pipeline {type(pipe)}"

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)

    device = pipe._execution_device

    # Prepare text embeddings
    if compute_embeddings_fn is not None:
        if is_sdxl:
            orig_size = [(1024, 1024)] * len(prompt)
            crop_coords = [(0, 0)] * len(prompt)
            encoded_text = compute_embeddings_fn(prompt, orig_size, crop_coords)
            prompt_embeds = encoded_text.pop("prompt_embeds")
        else:
            prompt_embeds = compute_embeddings_fn(prompt)["prompt_embeds"]
            encoded_text = {}
        prompt_embeds = prompt_embeds.to(pipe.unet.dtype)
    else:
        prompt_embeds = pipe.encode_prompt(prompt, device, 1, False)[0]
        encoded_text = {}
    assert prompt_embeds.dtype == pipe.unet.dtype

    # Prepare the DDIM solver
    endpoints = ','.join(['0'] + inverse_endpoints.split(',')[:-1]) if inverse_endpoints is not None else None
    solver = DDIMSolver(
        pipe.scheduler.alphas_cumprod.cpu().numpy(),
        timesteps=pipe.scheduler.num_train_timesteps,
        ddim_timesteps=num_scales,
        num_endpoints=num_inference_steps,
        num_inverse_endpoints=num_inference_steps,
        max_inverse_timestep_index=max_inverse_timestep_index,
        endpoints=endpoints,
        inverse_endpoints=inverse_endpoints
    ).to(device)

    if timesteps is None:
        timesteps = solver.inverse_endpoints.flip(0)
        boundary_timesteps = solver.endpoints.flip(0)
    else:
        timesteps, boundary_timesteps = timesteps, timesteps
        boundary_timesteps = boundary_timesteps[1:] + [boundary_timesteps[0]]
        boundary_timesteps[-1] = 999
        timesteps, boundary_timesteps = torch.tensor(timesteps), torch.tensor(boundary_timesteps)

    alpha_schedule = torch.sqrt(pipe.scheduler.alphas_cumprod).to(device)
    sigma_schedule = torch.sqrt(1 - pipe.scheduler.alphas_cumprod).to(device)

    # 5. Prepare latent variables
    num_channels_latents = pipe.unet.config.in_channels
    start_latents = pipe.prepare_latents(
        images, timesteps[0], batch_size, 1, prompt_embeds.dtype, device,
        generator=torch.Generator().manual_seed(seed),
    )
    latents = start_latents.clone()

    if guidance_scale is not None:
        w = torch.ones(batch_size) * guidance_scale
        w_embedding = guidance_scale_embedding(w, embedding_dim=512)
        w_embedding = w_embedding.to(device=latents.device, dtype=latents.dtype)
    else:
        w_embedding = None

    for i, (t, s) in enumerate(zip(timesteps, boundary_timesteps)):
        # predict the noise residual
        noise_pred = pipe.unet(
            latents.to(prompt_embeds.dtype),
            t,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
            timestep_cond=w_embedding,
            added_cond_kwargs=encoded_text,
        )[0]

        latents = predicted_origin(
            noise_pred,
            torch.tensor([t] * len(latents), device=device),
            torch.tensor([s] * len(latents), device=device),
            latents,
            pipe.scheduler.config.prediction_type,
            alpha_schedule,
            sigma_schedule,
        ).to(prompt_embeds.dtype)

    if return_start_latent:
        return latents, start_latents
    else:
        return latents


def linear_schedule_old(t, guidance_scale, tau1, tau2):
    t = t / 1000
    if t <= tau1:
        gamma = 1.0
    elif t >= tau2:
        gamma = 0.0
    else:
        gamma = (tau2 - t) / (tau2 - tau1)
    return gamma * guidance_scale


@torch.no_grad()
def sample_deterministic(
        pipe,
        prompt,
        latents=None,
        generator=None,
        num_scales=50,
        num_inference_steps=1,
        timesteps=None,
        start_timestep=19,
        max_inverse_timestep_index=49,
        return_latent=False,
        guidance_scale=None,  # Used only if the student has w_embedding
        compute_embeddings_fn=None,
        is_sdxl=False,
        endpoints=None,
        use_dynamic_guidance=False,
        tau1=0.7,
        tau2=0.7,
        amplify_prompt=None,
):
    # assert isinstance(pipe, StableDiffusionPipeline), f"Does not support the pipeline {type(pipe)}"
    height = pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = pipe.unet.config.sample_size * pipe.vae_scale_factor

    # 1. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)

    device = pipe._execution_device

    # Prepare text embeddings
    if compute_embeddings_fn is not None:
        if is_sdxl:
            orig_size = [(1024, 1024)] * len(prompt)
            crop_coords = [(0, 0)] * len(prompt)
            encoded_text = compute_embeddings_fn(prompt, orig_size, crop_coords)
            prompt_embeds = encoded_text.pop("prompt_embeds")
            if amplify_prompt is not None:
                orig_size = [(1024, 1024)] * len(amplify_prompt)
                crop_coords = [(0, 0)] * len(amplify_prompt)
                encoded_text_old = compute_embeddings_fn(amplify_prompt, orig_size, crop_coords)
                amplify_prompt_embeds = encoded_text_old.pop("prompt_embeds")
        else:
            prompt_embeds = compute_embeddings_fn(prompt)["prompt_embeds"]
            encoded_text = {}
        prompt_embeds = prompt_embeds.to(pipe.unet.dtype)
    else:
        prompt_embeds = pipe.encode_prompt(prompt, device, 1, False)[0]
        encoded_text = {}
    assert prompt_embeds.dtype == pipe.unet.dtype

    # Prepare the DDIM solver
    inverse_endpoints = ','.join(endpoints.split(',')[1:] + ['999']) if endpoints is not None else None
    solver = DDIMSolver(
        pipe.scheduler.alphas_cumprod.numpy(),
        timesteps=pipe.scheduler.num_train_timesteps,
        ddim_timesteps=num_scales,
        num_endpoints=num_inference_steps,
        num_inverse_endpoints=num_inference_steps,
        max_inverse_timestep_index=max_inverse_timestep_index,
        endpoints=endpoints,
        inverse_endpoints=inverse_endpoints
    ).to(device)

    prompt_embeds_init = copy.deepcopy(prompt_embeds)

    if timesteps is None:
        timesteps = solver.inverse_endpoints.flip(0)
        boundary_timesteps = solver.endpoints.flip(0)
    else:
        timesteps, boundary_timesteps = copy.deepcopy(timesteps), copy.deepcopy(timesteps)
        timesteps.reverse()
        boundary_timesteps.reverse()
        boundary_timesteps = boundary_timesteps[1:] + [boundary_timesteps[0]]
        boundary_timesteps[-1] = 0
        timesteps, boundary_timesteps = torch.tensor(timesteps), torch.tensor(boundary_timesteps)

    alpha_schedule = torch.sqrt(pipe.scheduler.alphas_cumprod).to(device)
    sigma_schedule = torch.sqrt(1 - pipe.scheduler.alphas_cumprod).to(device)

    # 5. Prepare latent variables
    if latents is None:
        num_channels_latents = pipe.unet.config.in_channels
        latents = pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            None,
        )
        assert latents.dtype == pipe.unet.dtype
    else:
        latents = latents.to(prompt_embeds.dtype)

    if guidance_scale is not None:
        w = torch.ones(batch_size) * guidance_scale
        w_embedding = guidance_scale_embedding(w, embedding_dim=512)
        w_embedding = w_embedding.to(device=latents.device, dtype=latents.dtype)
    else:
        w_embedding = None

    for i, (t, s) in enumerate(zip(timesteps, boundary_timesteps)):
        if use_dynamic_guidance:
            if not isinstance(t, int):
                t_item = t.item()
            if t_item > tau1 * 1000 and amplify_prompt is not None:
                prompt_embeds = amplify_prompt_embeds
            else:
                prompt_embeds = prompt_embeds_init
            guidance_scale = linear_schedule_old(t_item, w, tau1=tau1, tau2=tau2)
            guidance_scale_tensor = torch.tensor([guidance_scale] * len(latents))
            w_embedding = guidance_scale_embedding(guidance_scale_tensor, embedding_dim=512)
            w_embedding = w_embedding.to(device=latents.device, dtype=latents.dtype)

        # predict the noise residual
        noise_pred = pipe.unet(
            latents,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=None,
            return_dict=False,
            timestep_cond=w_embedding,
            added_cond_kwargs=encoded_text,
        )[0]

        latents = predicted_origin(
            noise_pred,
            torch.tensor([t] * len(noise_pred)).to(device),
            torch.tensor([s] * len(noise_pred)).to(device),
            latents,
            pipe.scheduler.config.prediction_type,
            alpha_schedule,
            sigma_schedule,
        ).to(pipe.unet.dtype)

    pipe.vae.to(torch.float32)
    image = pipe.vae.decode(latents.to(torch.float32) / pipe.vae.config.scaling_factor, return_dict=False)[0]
    do_denormalize = [True] * image.shape[0]
    image = pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=do_denormalize)

    if return_latent:
        return image, latents
    else:
        return image
# ------------------------------------------------------------------------