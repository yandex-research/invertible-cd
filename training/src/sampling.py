import torch

from src.lcm import DDIMSolver, guidance_scale_embedding, predicted_origin


@torch.no_grad()
def reverse_sample(
    pipe,
    prompt,
    latents=None,
    generator=None,
    num_scales=50,
    num_inference_steps=1,
    start_timestep=19,
    return_latent=False,
    guidance_scale=None,  # Used only if the student has w_embedding
    compute_embeddings_fn=None,
    is_sdxl=False,
    endpoints=None,
    vae_batch_size=2,
):
    """Sampling using reverse iCD"""
    height = pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = pipe.unet.config.sample_size * pipe.vae_scale_factor

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
    forward_endpoints = (
        ",".join(endpoints.split(",")[1:] + ["999"]) if endpoints is not None else None
    )
    solver = DDIMSolver(
        pipe.scheduler.alphas_cumprod.numpy(),
        timesteps=pipe.scheduler.num_train_timesteps,
        ddim_timesteps=num_scales,
        num_endpoints=num_inference_steps,
        num_forward_endpoints=num_inference_steps,
        endpoints=endpoints,
        forward_endpoints=forward_endpoints,
    ).to(device)

    timesteps = solver.forward_endpoints.flip(0)
    boundary_timesteps = solver.endpoints.flip(0)

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

    # Sampling using reverse iCD
    for t, s in zip(timesteps, boundary_timesteps):
        with torch.autocast("cuda", dtype=torch.float16):
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

    # Decode in batch mode to avoid OOM
    image = torch.cat(
        [
            pipe.vae.decode(
                latents[i : i + vae_batch_size] / pipe.vae.config.scaling_factor,
                return_dict=False,
            )[0]
            for i in range(0, latents.shape[0], vae_batch_size)
        ], dim=0,
    )

    do_denormalize = [True] * image.shape[0]
    image = pipe.image_processor.postprocess(
        image, output_type="pil", do_denormalize=do_denormalize
    )

    if return_latent:
        return image, latents
    else:
        return image


@torch.no_grad()
def forward_sample(
    pipe,
    images,
    prompt,
    num_scales=50,
    num_inference_steps=1,
    start_timestep=19, # start encoding from a slightly noised image
    return_start_latent=False,
    guidance_scale=None,  # Used only if the student has w_embedding
    compute_embeddings_fn=None,
    is_sdxl=False,
    forward_endpoints=None,
):
    """Sampling using forward iCD"""
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
    endpoints = (
        ",".join(["0"] + forward_endpoints.split(",")[:-1])
        if forward_endpoints is not None
        else None
    )
    solver = DDIMSolver(
        pipe.scheduler.alphas_cumprod.cpu().numpy(),
        timesteps=pipe.scheduler.num_train_timesteps,
        ddim_timesteps=num_scales,
        num_endpoints=num_inference_steps,
        num_forward_endpoints=num_inference_steps,
        endpoints=endpoints,
        forward_endpoints=forward_endpoints,
    ).to(device)

    timesteps, boundary_timesteps = solver.endpoints, solver.forward_endpoints
    timesteps[0] = start_timestep

    alpha_schedule = torch.sqrt(pipe.scheduler.alphas_cumprod).to(device)
    sigma_schedule = torch.sqrt(1 - pipe.scheduler.alphas_cumprod).to(device)

    # Prepare latent variables
    start_latents = pipe.prepare_latents(
        images, timesteps[0], batch_size, 1, prompt_embeds.dtype, device
    )
    latents = start_latents.clone()

    if guidance_scale is not None:
        w = torch.ones(batch_size) * guidance_scale
        w_embedding = guidance_scale_embedding(w, embedding_dim=512)
        w_embedding = w_embedding.to(device=latents.device, dtype=latents.dtype)
    else:
        w_embedding = None

    # Sampling using forward iCD
    for t, s in zip(timesteps, boundary_timesteps):
        with torch.autocast('cuda', dtype=torch.float16):
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
        )

    if return_start_latent:
        return latents, start_latents
    else:
        return latents
