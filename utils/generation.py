import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from typing import Union
from IPython.display import display
from utils import p2p


# Main function to run
# ----------------------------------------------------------------------
@torch.no_grad()
def runner(
        model,
        prompt,
        controller,
        solver,
        is_cons_forward=False,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=None,
        latent=None,
        uncond_embeddings=None,
        start_time=50,
        return_type='image',
        dynamic_guidance=False,
        tau1=0.4,
        tau2=0.6,
        w_embed_dim=0,
):
    p2p.register_attention_control(model, controller)
    height = width = 512
    solver.init_prompt(prompt, None)
    latent, latents = init_latent(latent, model, 512, 512, generator, len(prompt))
    model.scheduler.set_timesteps(num_inference_steps)
    dynamic_guidance = True if tau1 < 1.0 or tau1 < 1.0 else False

    if not is_cons_forward:
        latents = solver.ddim_loop(latents,
                                   num_inference_steps,
                                   is_forward=False,
                                   guidance_scale=guidance_scale,
                                   dynamic_guidance=dynamic_guidance,
                                   tau1=tau1,
                                   tau2=tau2,
                                   w_embed_dim=w_embed_dim,
                                   uncond_embeddings=uncond_embeddings if uncond_embeddings is not None else None,
                                   controller=controller)
        latents = latents[-1]
    else:
        latents = solver.cons_generation(
            latents,
            guidance_scale=guidance_scale,
            w_embed_dim=w_embed_dim,
            dynamic_guidance=dynamic_guidance,
            tau1=tau1,
            tau2=tau2,
            controller=controller)
        latents = latents[-1]

    if return_type == 'image':
        image = latent2image(model.vae, latents.to(model.vae.dtype))
    else:
        image = latents

    return image, latent


# ----------------------------------------------------------------------


# Utils
# ----------------------------------------------------------------------
def linear_schedule_old(t, guidance_scale, tau1, tau2):
    t = t / 1000
    if t <= tau1:
        gamma = 1.0
    elif t >= tau2:
        gamma = 0.0
    else:
        gamma = (tau2 - t) / (tau2 - tau1)
    return gamma * guidance_scale


def linear_schedule(t, guidance_scale, tau1=0.4, tau2=0.8):
    t = t / 1000
    if t <= tau1:
        return guidance_scale
    if t >= tau2:
        return 1.0
    gamma = (tau2 - t) / (tau2 - tau1) * (guidance_scale - 1.0) + 1.0

    return gamma


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


# ----------------------------------------------------------------------


# Diffusion step with scheduler from diffusers and controller for editing
# ----------------------------------------------------------------------
def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def predicted_origin(model_output, timesteps, boundary_timesteps, sample, prediction_type, alphas, sigmas):
    sigmas_s = extract_into_tensor(sigmas, boundary_timesteps, sample.shape)
    alphas_s = extract_into_tensor(alphas, boundary_timesteps, sample.shape)

    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)

    # Set hard boundaries to ensure equivalence with forward (direct) CD
    alphas_s[boundary_timesteps == 0] = 1.0
    sigmas_s[boundary_timesteps == 0] = 0.0

    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas  # x0 prediction
        pred_x_0 = alphas_s * pred_x_0 + sigmas_s * model_output  # Euler step to the boundary step
    elif prediction_type == "v_prediction":
        assert boundary_timesteps == 0, "v_prediction does not support multiple endpoints at the moment"
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")
    return pred_x_0


def guided_step(noise_prediction_text,
                noise_pred_uncond,
                t,
                guidance_scale,
                dynamic_guidance=False,
                tau1=0.4,
                tau2=0.6):
    if dynamic_guidance:
        if not isinstance(t, int):
            t = t.item()
        new_guidance_scale = linear_schedule(t, guidance_scale, tau1=tau1, tau2=tau2)
    else:
        new_guidance_scale = guidance_scale

    noise_pred = noise_pred_uncond + new_guidance_scale * (noise_prediction_text - noise_pred_uncond)
    return noise_pred


# ----------------------------------------------------------------------


# DDIM scheduler with inversion
# ----------------------------------------------------------------------
class Generator:

    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self,
                       model,
                       latent,
                       t,
                       guidance_scale=1,
                       context=None,
                       w_embed_dim=0,
                       dynamic_guidance=False,
                       tau1=0.4,
                       tau2=0.6):
        latents_input = torch.cat([latent] * 2)
        if context is None:
            context = self.context

        # w embed 
        # --------------------------------------
        if w_embed_dim > 0:
            if dynamic_guidance:
                if not isinstance(t, int):
                    t_item = t.item()
                guidance_scale = linear_schedule_old(t_item, guidance_scale, tau1=tau1, tau2=tau2)  # TODO UPDATE
            if len(latents_input) == 4:
                guidance_scale_tensor = torch.tensor([0.0, 0.0, 0.0, guidance_scale])
            else:
                guidance_scale_tensor = torch.tensor([guidance_scale] * len(latents_input))
            w_embedding = guidance_scale_embedding(guidance_scale_tensor, embedding_dim=w_embed_dim)
            w_embedding = w_embedding.to(device=latent.device, dtype=latent.dtype)
        else:
            w_embedding = None
        # --------------------------------------
        noise_pred = model.unet(latents_input.to(dtype=model.unet.dtype),
                                t,
                                timestep_cond=w_embedding.to(dtype=model.unet.dtype) if w_embed_dim > 0 else None,
                                encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)

        if guidance_scale > 1 and w_embedding is None:
            noise_pred = guided_step(noise_prediction_text, noise_pred_uncond, t, guidance_scale, dynamic_guidance,
                                     tau1, tau2)
        else:
            noise_pred = noise_prediction_text

        return noise_pred

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents.to(dtype=self.model.dtype))['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            elif type(image) is list:
                image = [np.array(i).reshape(1, 512, 512, 3) for i in image]
                image = np.concatenate(image)
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(0, 3, 1, 2).to(self.model.device, dtype=self.model.vae.dtype)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(self.model.device, dtype=self.model.dtype)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt, uncond_embeddings=None):
        if uncond_embeddings is None:
            uncond_input = self.model.tokenizer(
                [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
                return_tensors="pt"
            )
            uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings.expand(*text_embeddings.shape), text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self,
                  latent,
                  n_steps,
                  is_forward=True,
                  guidance_scale=1,
                  dynamic_guidance=False,
                  tau1=0.4,
                  tau2=0.6,
                  w_embed_dim=0,
                  uncond_embeddings=None,
                  controller=None):
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in tqdm(range(n_steps)):
            if uncond_embeddings is not None:
                self.init_prompt(self.prompt, uncond_embeddings[i])
            if is_forward:
                t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            else:
                t = self.model.scheduler.timesteps[i]
            noise_pred = self.get_noise_pred(
                model=self.model,
                latent=latent,
                t=t,
                context=None,
                guidance_scale=guidance_scale,
                dynamic_guidance=dynamic_guidance,
                w_embed_dim=w_embed_dim,
                tau1=tau1,
                tau2=tau2)
            if is_forward:
                latent = self.next_step(noise_pred, t, latent)
            else:
                latent = self.prev_step(noise_pred, t, latent)
            if controller is not None:
                latent = controller.step_callback(latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self,
                       image,
                       n_steps=None,
                       guidance_scale=1,
                       dynamic_guidance=False,
                       tau1=0.4,
                       tau2=0.6,
                       w_embed_dim=0):

        if n_steps is None:
            n_steps = self.n_steps
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent,
                                      is_forward=True,
                                      guidance_scale=guidance_scale,
                                      n_steps=n_steps,
                                      dynamic_guidance=dynamic_guidance,
                                      tau1=tau1,
                                      tau2=tau2,
                                      w_embed_dim=w_embed_dim)
        return image_rec, ddim_latents

    @torch.no_grad()
    def cons_generation(self,
                        latent,
                        guidance_scale=1,
                        dynamic_guidance=False,
                        tau1=0.4,
                        tau2=0.6,
                        w_embed_dim=0,
                        controller=None, ):

        all_latent = [latent]
        latent = latent.clone().detach()
        alpha_schedule = torch.sqrt(self.model.scheduler.alphas_cumprod).to(self.model.device)
        sigma_schedule = torch.sqrt(1 - self.model.scheduler.alphas_cumprod).to(self.model.device)

        for i, (t, s) in enumerate(tqdm(zip(self.reverse_timesteps, self.reverse_boundary_timesteps))):
            noise_pred = self.get_noise_pred(
                model=self.reverse_cons_model,
                latent=latent,
                t=t.to(self.model.device),
                context=None,
                tau1=tau1, tau2=tau2,
                w_embed_dim=w_embed_dim,
                guidance_scale=guidance_scale,
                dynamic_guidance=dynamic_guidance)

            latent = predicted_origin(
                noise_pred,
                torch.tensor([t] * len(latent), device=self.model.device),
                torch.tensor([s] * len(latent), device=self.model.device),
                latent,
                self.model.scheduler.config.prediction_type,
                alpha_schedule,
                sigma_schedule,
            )
            if controller is not None:
                latent = controller.step_callback(latent)
            all_latent.append(latent)

        return all_latent

    @torch.no_grad()
    def cons_inversion(self,
                       image,
                       guidance_scale=0.0,
                       w_embed_dim=0,
                       seed=0):
        alpha_schedule = torch.sqrt(self.model.scheduler.alphas_cumprod).to(self.model.device)
        sigma_schedule = torch.sqrt(1 - self.model.scheduler.alphas_cumprod).to(self.model.device)

        # 5. Prepare latent variables
        latent = self.image2latent(image)
        generator = torch.Generator().manual_seed(seed)
        noise = torch.randn(latent.shape, generator=generator).to(latent.device)
        latent = self.noise_scheduler.add_noise(latent, noise, torch.tensor([self.start_timestep]))
        image_rec = self.latent2image(latent)

        for i, (t, s) in enumerate(tqdm(zip(self.forward_timesteps, self.forward_boundary_timesteps))):
            # predict the noise residual
            noise_pred = self.get_noise_pred(
                model=self.forward_cons_model,
                latent=latent,
                t=t.to(self.model.device),
                context=None,
                guidance_scale=guidance_scale,
                w_embed_dim=w_embed_dim,
                dynamic_guidance=False)

            latent = predicted_origin(
                noise_pred,
                torch.tensor([t] * len(latent), device=self.model.device),
                torch.tensor([s] * len(latent), device=self.model.device),
                latent,
                self.model.scheduler.config.prediction_type,
                alpha_schedule,
                sigma_schedule,
            )

        return image_rec, [latent]

    def _create_forward_inverse_timesteps(self,
                                          num_endpoints,
                                          n_steps,
                                          max_inverse_timestep_index):
        timestep_interval = n_steps // num_endpoints + int(n_steps % num_endpoints > 0)
        endpoint_idxs = torch.arange(timestep_interval, n_steps, timestep_interval) - 1
        inverse_endpoint_idxs = torch.arange(timestep_interval, n_steps, timestep_interval) - 1
        inverse_endpoint_idxs = torch.tensor(inverse_endpoint_idxs.tolist() + [max_inverse_timestep_index])

        endpoints = torch.tensor([0] + self.ddim_timesteps[endpoint_idxs].tolist())
        inverse_endpoints = self.ddim_timesteps[inverse_endpoint_idxs]

        return endpoints, inverse_endpoints

    def __init__(self,
                 model,
                 n_steps,
                 noise_scheduler,
                 forward_cons_model=None,
                 reverse_cons_model=None,
                 num_endpoints=1,
                 num_forward_endpoints=1,
                 reverse_timesteps=None,
                 forward_timesteps=None,
                 max_forward_timestep_index=49,
                 start_timestep=19):

        self.model = model
        self.forward_cons_model = forward_cons_model
        self.reverse_cons_model = reverse_cons_model
        self.noise_scheduler = noise_scheduler

        self.n_steps = n_steps
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(n_steps)
        self.prompt = None
        self.context = None
        step_ratio = 1000 // n_steps
        self.ddim_timesteps = (np.arange(1, n_steps + 1) * step_ratio).round().astype(np.int64) - 1
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.start_timestep = start_timestep

        # Set endpoints for direct CTM
        if reverse_timesteps is None or forward_timesteps is None:
            endpoints, inverse_endpoints = self._create_forward_inverse_timesteps(num_endpoints, n_steps,
                                                                                  max_forward_timestep_index)
            self.reverse_timesteps, self.reverse_boundary_timesteps = inverse_endpoints.flip(0), endpoints.flip(0)

            # Set endpoints for forward CTM
            endpoints, inverse_endpoints = self._create_forward_inverse_timesteps(num_forward_endpoints, n_steps,
                                                                                  max_forward_timestep_index)
            self.forward_timesteps, self.forward_boundary_timesteps = endpoints, inverse_endpoints
            self.forward_timesteps[0] = self.start_timestep
        else:
            self.reverse_timesteps, self.reverse_boundary_timesteps = reverse_timesteps, reverse_timesteps
            self.reverse_timesteps.reverse()
            self.reverse_boundary_timesteps = self.reverse_boundary_timesteps[1:] + [self.reverse_boundary_timesteps[0]]
            self.reverse_boundary_timesteps[-1] = 0
            self.reverse_timesteps, self.reverse_boundary_timesteps = torch.tensor(reverse_timesteps), torch.tensor(
                self.reverse_boundary_timesteps)

            self.forward_timesteps, self.forward_boundary_timesteps = forward_timesteps, forward_timesteps
            self.forward_boundary_timesteps = self.forward_boundary_timesteps[1:] + [self.forward_boundary_timesteps[0]]
            self.forward_boundary_timesteps[-1] = 999
            self.forward_timesteps, self.forward_boundary_timesteps = torch.tensor(
                self.forward_timesteps), torch.tensor(self.forward_boundary_timesteps)

        print(f"Endpoints reverse CTM: {self.reverse_timesteps}, {self.reverse_boundary_timesteps}")
        print(f"Endpoints forward CTM: {self.forward_timesteps}, {self.forward_boundary_timesteps}")

# ----------------------------------------------------------------------

# 3rd party utils
# ----------------------------------------------------------------------
def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size, model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


def load_512(image_path, left=0, right=0, top=0, bottom=0):
    # if type(image_path) is str:
    #    image = np.array(Image.open(image_path))[:, :, :3]
    # else:
    #    image = image_path
    # h, w, c = image.shape
    # left = min(left, w - 1)
    # right = min(right, w - left - 1)
    # top = min(top, h - left - 1)
    # bottom = min(bottom, h - top - 1)
    # image = image[top:h - bottom, left:w - right]
    # h, w, c = image.shape
    # if h < w:
    #    offset = (w - h) // 2
    #    image = image[:, offset:offset + h]
    # elif w < h:
    #    offset = (h - w) // 2
    #    image = image[offset:offset + w]
    image = np.array(Image.open(image_path).convert('RGB'))[:, :, :3]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


def to_pil_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    return pil_img


def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    display(pil_img)
# ----------------------------------------------------------------------
