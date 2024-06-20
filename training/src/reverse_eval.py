import gc
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from accelerate.logging import get_logger
from torchvision.transforms import ToPILImage
from tqdm.auto import tqdm

from src.sampling import reverse_sample

logger = get_logger(__name__)

from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from peft import get_peft_model_state_dict


def get_module_kohya_state_dict(
    module, prefix: str, dtype: torch.dtype, adapter_name: str = "default"
):
    kohya_ss_state_dict = {}
    for peft_key, weight in get_peft_model_state_dict(
        module, adapter_name=adapter_name
    ).items():
        kohya_key = peft_key.replace("base_model.model", prefix)
        kohya_key = kohya_key.replace("lora_A", "lora_down")
        kohya_key = kohya_key.replace("lora_B", "lora_up")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_ss_state_dict[kohya_key] = weight.to(dtype)

        # Set alpha parameter
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = torch.tensor(
                module.peft_config[adapter_name].lora_alpha
            ).to(dtype)

    return kohya_ss_state_dict


def load_sd_pipeline(args, unet, vae, weight_dtype, device="cuda", is_sdxl=False):
    """Load StableDiffusion pipeline given already loaded UNet and VAE"""
    pipeline_class = StableDiffusionXLPipeline if is_sdxl else StableDiffusionPipeline

    if isinstance(unet, UNet2DConditionModel):
        pipeline = pipeline_class.from_pretrained(
            args.pretrained_teacher_model,
            unet=unet,
            vae=vae,
            scheduler=DDIMScheduler.from_pretrained(
                args.pretrained_teacher_model, subfolder="scheduler"
            ),
            revision=args.revision,
            torch_dtype=weight_dtype,
            safety_checker=None,
        )
    else:
        # Extract LoRA parameters from unet
        lora_state_dict = get_module_kohya_state_dict(unet, "lora_unet", weight_dtype)
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
        pipeline.load_lora_weights(lora_state_dict)
        pipeline.fuse_lora()

    pipeline.set_progress_bar_config(disable=True)
    return pipeline.to(device)


@torch.no_grad()
def log_validation(
    vae,
    unet,
    args,
    accelerator,
    weight_dtype,
    step,
    is_sdxl=False,
    compute_embeddings_fn=None,
):
    """Generate a few samples for visualization"""

    logger.info("Running validation... ")
    unet = accelerator.unwrap_model(unet)
    pipeline = load_sd_pipeline(
        args, unet, vae, weight_dtype, device=accelerator.device, is_sdxl=is_sdxl
    )

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # Init eval guidance scale
    if args.discrete_w is not None:
        guidance_scale = 7.0 # by default, perform evaluation for CFG=7.0 (8.0 in the classical formulation). TODO: add to args.
    elif args.embed_guidance:
        guidance_scale = (args.w_max + args.w_min) / 2
    else:
        guidance_scale = None
    logger.info(f"Guidance scale: {guidance_scale}")

    validation_prompts = [
        "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
        "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
        "A sad puppy with large eyes",
        "A girl with pale blue hair and a cami tank top",
        "cute girl, Kyoto animation, 4k, high resolution",
        "A person laying on a surfboard holding his dog",
        "Green commercial building with refrigerator and refrigeration units outside",
        "An airplane with two propellor engines flying in the sky",
        "Four cows in a pen on a sunny day",
        "Three dogs sleeping together on an unmade bed",
        "a deer with bird feathers, highly detailed, full body",
    ]

    image_logs = []
    for _, prompt in enumerate(validation_prompts):
        images = reverse_sample(
            pipeline,
            [prompt] * 4,
            num_inference_steps=args.num_endpoints,
            generator=generator,
            guidance_scale=guidance_scale,
            is_sdxl=is_sdxl,
            compute_embeddings_fn=compute_embeddings_fn,
            endpoints=args.endpoints,
        )
        image_logs.append({"validation_prompt": prompt, "images": images})

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = (
                    log["validation_prompt"] + f" | CFG {guidance_scale + 1}"
                )
                formatted_images = []
                for image in images:
                    formatted_images.append(np.asarray(image.resize((256, 256))))

                formatted_images = np.stack(formatted_images)
                tracker.writer.add_images(
                    validation_prompt, formatted_images, step, dataformats="NHWC"
                )
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return image_logs


def prepare_val_prompts(path, bs=20, max_cnt=5000):
    """Load the prompts for metric evaluation"""
    df = pd.read_csv(path)
    all_text = list(df["caption"])
    all_text = all_text[:max_cnt]

    num_batches = (
        (len(all_text) - 1) // (bs * dist.get_world_size()) + 1
    ) * dist.get_world_size()
    all_batches = np.array_split(np.array(all_text), num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    index_list = np.arange(len(all_text))
    all_batches_index = np.array_split(index_list, num_batches)
    rank_batches_index = all_batches_index[dist.get_rank() :: dist.get_world_size()]
    return rank_batches, rank_batches_index, all_text


@torch.no_grad()
def distributed_sampling(
    vae,
    unet,
    args,
    accelerator,
    weight_dtype,
    step,
    num_inference_steps=4,
    batch_size=4,
    max_cnt=5000,
    is_sdxl=False,
    compute_embeddings_fn=None,
):
    """Sampling many images in parallel. Used for metric evaluation"""
    logger.info("Running sampling...")
    unet = accelerator.unwrap_model(unet)
    pipeline = load_sd_pipeline(
        args, unet, vae, weight_dtype, device=accelerator.device, is_sdxl=is_sdxl
    )

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # Init eval guidance scale
    guidance_scale = 7.0 if args.embed_guidance else None
    logger.info(f"Reverse guidance scale: {guidance_scale}")

    # Prepare validation prompts
    rank_batches, rank_batches_index, prompts = prepare_val_prompts(
        args.val_prompt_path, bs=batch_size, max_cnt=max_cnt
    )

    all_images, all_prompts = [], []
    for cnt, mini_batch in enumerate(
        tqdm(rank_batches, unit="batch", disable=(dist.get_rank() != 0))
    ):
        local_images = []
        local_text_idxs = []
        images = reverse_sample(
            pipeline,
            list(mini_batch),
            num_inference_steps=args.num_endpoints,
            generator=generator,
            guidance_scale=guidance_scale,
            is_sdxl=is_sdxl,
            compute_embeddings_fn=compute_embeddings_fn,
            endpoints=args.endpoints,
        )

        for text_idx, global_idx in enumerate(rank_batches_index[cnt]):
            img_tensor = torch.tensor(np.array(images[text_idx]))
            local_images.append(img_tensor)
            local_text_idxs.append(global_idx)

        local_images = torch.stack(local_images).cuda()
        local_text_idxs = torch.tensor(local_text_idxs).cuda()

        gathered_images = [
            torch.zeros_like(local_images) for _ in range(dist.get_world_size())
        ]
        gathered_text_idxs = [
            torch.zeros_like(local_text_idxs) for _ in range(dist.get_world_size())
        ]

        dist.all_gather(gathered_images, local_images)  # gather not supported with NCCL
        dist.all_gather(gathered_text_idxs, local_text_idxs)

        if accelerator.is_main_process:
            gathered_images = np.concatenate(
                [images.cpu().numpy() for images in gathered_images], axis=0
            )
            gathered_text_idxs = np.concatenate(
                [text_idxs.cpu().numpy() for text_idxs in gathered_text_idxs], axis=0
            )
            for image, global_idx in zip(gathered_images, gathered_text_idxs):
                all_images.append(ToPILImage()(image))
                all_prompts.append(prompts[global_idx])
    # Done.
    dist.barrier()
    return all_images, all_prompts
