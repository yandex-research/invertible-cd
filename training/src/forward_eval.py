import gc

import numpy as np
import torch
import torch.distributed as dist
from accelerate.logging import get_logger
from torchvision.transforms import ToPILImage
from tqdm.auto import tqdm

from src.datasets import get_coco_loader
from src.reverse_eval import get_module_kohya_state_dict, load_sd_pipeline
from src.fid_score_in_memory import calculate_fid
from src.sampling import forward_sample, reverse_sample
from src.utils import to_image

logger = get_logger(__name__)

from diffusers import (
    DDIMScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
    UNet2DConditionModel,
)


def load_forward_sd_pipeline(
    args, unet, vae, weight_dtype, device="cuda", is_sdxl=False
):
    # Extract LoRA parameters from unet
    lora_state_dict = get_module_kohya_state_dict(unet, "lora_unet", weight_dtype)

    # Create the teacher UNet
    init_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="unet",
        revision=args.teacher_revision,
        time_cond_proj_dim=512 if args.embed_guidance else None,
        low_cpu_mem_usage=False,
        device_map=None,
    )
    if args.cfg_distill_teacher_path:
        logger.info(f"Loading teacher from {args.cfg_distill_teacher_path}")
        state_dict = torch.load(args.cfg_distill_teacher_path, map_location="cpu")
        init_unet.load_state_dict(state_dict)

    # Create the pipeline with the teacher UNet
    pipeline_class = (
        StableDiffusionXLImg2ImgPipeline if is_sdxl else StableDiffusionImg2ImgPipeline
    )
    pipeline = pipeline_class.from_pretrained(
        args.pretrained_teacher_model,
        unet=init_unet,
        vae=vae,
        scheduler=DDIMScheduler.from_pretrained(
            args.pretrained_teacher_model, subfolder="scheduler"
        ),
        revision=args.revision,
        torch_dtype=weight_dtype,
        safety_checker=None,
    )
    pipeline.set_progress_bar_config(disable=True)

    # Load and fuse the forward CD LoRA
    pipeline.load_lora_weights(lora_state_dict)
    pipeline.fuse_lora()
    return pipeline.to(device)


@torch.no_grad()
def log_validation_inversion(
    vae,
    unet,
    forward_unet,
    args,
    accelerator,
    forward_accelerator,
    weight_dtype,
    step,
    num_validation_prompts=8,
    is_sdxl=False,
    compute_embeddings_fn=None,
    vae_batch_size=2,
):
    logger.info("Running validation for inversion... ")

    forward_unet = forward_accelerator.unwrap_model(forward_unet)
    forward_pipeline = load_forward_sd_pipeline(
        args,
        forward_unet,
        vae,
        weight_dtype,
        device=accelerator.device,
        is_sdxl=is_sdxl,
    )

    validation_data = get_coco_loader(args, batch_size=num_validation_prompts)
    batch = next(validation_data)

    # Init eval guidance scale
    assert args.forward_w_max == args.forward_w_min == 0
    forward_guidance_scale = (
        0.0 if args.embed_guidance else None
    )  # Evaluate inversion only for unguided processes in both directions
    logger.info(f"forward guidance scale: {forward_guidance_scale}")

    # Prepare latents using forward CM
    batch["latents"] = forward_sample(
        forward_pipeline,
        batch["image"],
        batch["text"],
        num_inference_steps=args.num_forward_endpoints,
        start_timestep=args.start_forward_timestep,
        guidance_scale=forward_guidance_scale,
        is_sdxl=is_sdxl,
        compute_embeddings_fn=compute_embeddings_fn,
        forward_endpoints=args.forward_endpoints,
    )
    # Decode in batch mode to avoid OOM
    decoded_latents = torch.cat(
        [
            vae.decode(
                batch["latents"][i : i + vae_batch_size].to(vae.dtype)
                / vae.config.scaling_factor,
                return_dict=False,
            )[0].cpu().float()
            for i in range(0, len(batch["latents"]), vae_batch_size)
        ], dim=0,
    )

    decoded_latents = forward_pipeline.image_processor.postprocess(
        decoded_latents,
        output_type="pil",
        do_denormalize=[True] * decoded_latents.shape[0],
    )

    # Prepare reverse CM pipeline (noise latents -> images)
    unet = accelerator.unwrap_model(unet)
    pipeline = load_sd_pipeline(
        args, unet, vae, weight_dtype, device=accelerator.device, is_sdxl=is_sdxl
    )

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    guidance_scale = (
        0.0 if args.embed_guidance else None
    )  # Evaluate inversion only for CFG=1 in both directions
    logger.info(f"reverse guidance scale: {guidance_scale}")

    # Reconstruct images from predicted noise latents
    cm_reconstructed_images = reverse_sample(
        pipeline,
        batch["text"],
        latents=batch["latents"],
        num_inference_steps=args.num_endpoints,
        generator=generator,
        guidance_scale=guidance_scale,
        is_sdxl=is_sdxl,
        compute_embeddings_fn=compute_embeddings_fn,
        endpoints=args.endpoints,
    )

    # Log latents and images
    image_logs = []
    for i in range(len(batch["image"])):
        logs = {
            "prompt": batch["text"][i],
            "orig_image": to_image(batch["image"][i]).resize((256, 256)),
            "decoded_latent": decoded_latents[i].resize((256, 256)),
            f"image_{args.num_endpoints}_steps": cm_reconstructed_images[i].resize((256, 256)),
        }
        image_logs.append(logs)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                validation_prompt = (
                    "INVERSION: " + log["prompt"] + f" | CFG {guidance_scale + 1}"
                )
                formatted_images = []
                formatted_images.append(np.asarray(log["decoded_latent"]))
                formatted_images.append(np.asarray(log["orig_image"]))
                formatted_images.append(
                    np.asarray(log[f"image_{args.num_endpoints}_steps"])
                )
                formatted_images = np.stack(formatted_images)
                tracker.writer.add_images(
                    validation_prompt, formatted_images, step, dataformats="NHWC"
                )
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    del pipeline
    del forward_pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return image_logs


def recon_loss_fn(predict, target):
    loss = (predict - target) ** 2
    loss_mean = loss.mean()
    return loss_mean


@torch.no_grad()
def eval_inversion(
    vae,
    unet,
    forward_unet,
    args,
    forward_accelerator,
    accelerator,
    weight_dtype,
    step,
    batch_size=4,
    is_sdxl=False,
    compute_embeddings_fn=None,
    max_cnt=5000,
):
    """Calculate metrics for the forward CD using the reverse CD"""
    logger.info("Running the evaluation of forward CD... ")
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    validation_data = get_coco_loader(args, batch_size=batch_size, max_cnt=max_cnt)

    forward_unet = forward_accelerator.unwrap_model(forward_unet)
    forward_pipeline = load_forward_sd_pipeline(
        args,
        forward_unet,
        vae,
        weight_dtype,
        device=accelerator.device,
        is_sdxl=is_sdxl,
    )

    unet = accelerator.unwrap_model(unet)
    reverse_pipeline = load_sd_pipeline(
        args, unet, vae, weight_dtype, device=accelerator.device, is_sdxl=is_sdxl
    )

    # Init eval guidance scale
    guidance_scale = (
        0.0 if args.embed_guidance else None
    )  # Evaluate inversion only for CFG=1 in both directions
    logger.info(f"reverse guidance scale: {guidance_scale}")

    assert args.forward_w_max == args.forward_w_min == 0
    forward_guidance_scale = (
        0.0 if args.embed_guidance else None
    )  # Evaluate inversion only for unguided processes in both directions
    logger.info(f"forward guidance scale: {forward_guidance_scale}")

    # Eval fid and reconstruction error
    recon_loss_cm = []
    all_images = []

    for batch in tqdm(validation_data, disable=(dist.get_rank() != 0)):

        # forward predict
        batch["latents"], start_latents = forward_sample(
            forward_pipeline,
            batch["image"],
            batch["text"],
            num_inference_steps=args.num_forward_endpoints,
            start_timestep=args.start_forward_timestep,
            num_scales=args.num_ddim_timesteps,
            return_start_latent=True,
            guidance_scale=forward_guidance_scale,
            is_sdxl=is_sdxl,
            compute_embeddings_fn=compute_embeddings_fn,
            forward_endpoints=args.forward_endpoints,
        )

        # reverse reconstruction
        cm_recon_images, cm_recon_latents = reverse_sample(
            reverse_pipeline,
            batch["text"],
            latents=batch["latents"],
            num_inference_steps=args.num_endpoints,
            start_timestep=args.start_forward_timestep,
            generator=generator,
            num_scales=args.num_ddim_timesteps,
            return_latent=True,
            guidance_scale=guidance_scale,
            is_sdxl=is_sdxl,
            compute_embeddings_fn=compute_embeddings_fn,
            endpoints=args.endpoints,
        )

        # Recon loss for iCD
        recon_loss_cm.append(
            recon_loss_fn(
                cm_recon_latents.cpu().float(), start_latents.cpu().float()
            ).item()
        )
        # Gather images
        local_images_cm = torch.tensor(
            np.concatenate(
                [np.expand_dims(np.array(item), axis=0) for item in cm_recon_images]
            )
        ).cuda()
        gathered_images_cm = [
            torch.zeros_like(local_images_cm) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            gathered_images_cm, local_images_cm
        )  # gather not supported with NCCL

        if accelerator.is_main_process:
            gathered_images_cm = np.concatenate(
                [images.cpu().numpy() for images in gathered_images_cm], axis=0
            )
            all_images.extend([ToPILImage()(image) for image in gathered_images_cm])

    # Gather recon losses
    recon_loss_cm = torch.tensor(recon_loss_cm).cuda()
    gathered_cm_losses = [
        torch.zeros_like(recon_loss_cm) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(gathered_cm_losses, recon_loss_cm)

    # Calculate metrics
    if accelerator.is_main_process:
        gathered_cm_losses = np.concatenate(
            [images.cpu().numpy() for images in gathered_cm_losses], axis=0
        )
        fid_score_cm = calculate_fid(
            all_images, args.coco_ref_stats_path, inception_path=args.inception_path
        )

        logs = {
            "recon_loss_cm": float(np.mean(gathered_cm_losses)),
            "fid_score_cm": fid_score_cm,
        }
        accelerator.log(logs, step=step)
    dist.barrier()
