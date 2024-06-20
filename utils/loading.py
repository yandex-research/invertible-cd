import torch
import copy
import pandas as pd

from safetensors.torch import load_file
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel, StableDiffusionXLPipeline, \
    StableDiffusionXLImg2ImgPipeline


def get_module_kohya_state_dict(module, prefix: str, dtype: torch.dtype, adapter_name: str = "default"):
    kohya_ss_state_dict = {}
    for peft_key, weight in module.items():
        kohya_key = peft_key.replace("unet.base_model.model", prefix)
        kohya_key = kohya_key.replace("lora_A", "lora_down")
        kohya_key = kohya_key.replace("lora_B", "lora_up")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_ss_state_dict[kohya_key] = weight.to(dtype)
        # Set alpha parameter
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = torch.tensor(8).to(dtype)

    return kohya_ss_state_dict


# Load models (DM, CM)
def load_models(model_id,
                device,
                reverse_checkpoint,
                forward_checkpoint,
                r=64,
                w_embed_dim=0,
                teacher_checkpoint=None,
                dtype='fp32',
                ):
    # Diffusion
    # ------------------------------------------------------------
    dtype = torch.float32 if dtype == 'fp32' else torch.float16
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                              set_alpha_to_one=False)
    ldm_stable = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler).to(device, dtype=dtype)
    # ------------------------------------------------------------

    # Reverse consistency
    # ------------------------------------------------------------
    if w_embed_dim > 0:
        print(f'Forward CD is initialized with guidance embedding, dim {w_embed_dim}')
        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet",
            time_cond_proj_dim=w_embed_dim, low_cpu_mem_usage=False, device_map=None
        ).to(device)
        if teacher_checkpoint is not None:
            print(f'Embedded model is loading from {teacher_checkpoint}')
            unet.load_state_dict(torch.load(teacher_checkpoint))
            ldm_stable.unet = copy.deepcopy(unet)
            ldm_stable.to(dtype=dtype)
        else:
            print('PROVIDE TEACHER')
    else:
        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet"
        ).to(device)

    if reverse_checkpoint is not None:
        print(f'Reverse CD is loading from {reverse_checkpoint}')
        reverse_cons_model = copy.deepcopy(ldm_stable)
        lora_weight = load_file(reverse_checkpoint)
        lora_state_dict = get_module_kohya_state_dict(lora_weight, "lora_unet", torch.float16)
        reverse_cons_model.load_lora_weights(lora_state_dict)
        reverse_cons_model.fuse_lora()
        reverse_cons_model.to(dtype=dtype)
    else:
        reverse_cons_model = None
    # ------------------------------------------------------------

    # Inverse consistency
    # ------------------------------------------------------------
    if forward_checkpoint is not None:
        print(f'Forward CD is loading from {forward_checkpoint}')
        forward_cons_model = copy.deepcopy(ldm_stable)
        lora_weight = load_file(forward_checkpoint)
        lora_state_dict = get_module_kohya_state_dict(lora_weight, "lora_unet", torch.float16)
        forward_cons_model.load_lora_weights(lora_state_dict)
        forward_cons_model.fuse_lora()
        forward_cons_model.to(dtype=dtype)
    else:
        forward_cons_model = None
    # ------------------------------------------------------------

    return ldm_stable, reverse_cons_model, forward_cons_model


def load_models_xl(model_id,
                   reverse_checkpoint,
                   forward_checkpoint,
                   teacher_checkpoint,
                   ):
    # Reverse consistency
    # ------------------------------------------------------------
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet",
        time_cond_proj_dim=512, low_cpu_mem_usage=False, device_map=None
    ).to(torch.float16)

    teacher_checkpoint = torch.load(
        teacher_checkpoint,
        map_location='cpu')
    unet.load_state_dict(teacher_checkpoint)

    stable_pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        unet=unet,
        scheduler=DDIMScheduler.from_pretrained(model_id, subfolder="scheduler"),
        safety_checker=None,
        variant='fp16',
        torch_dtype=torch.float16
    ).to('cuda')

    lora_weight = load_file(reverse_checkpoint)

    print(f'Reverse CD is loading from {reverse_checkpoint}')
    lora_state_dict = get_module_kohya_state_dict(lora_weight, "lora_unet", torch.float32)
    pipe = copy.deepcopy(stable_pipe)
    pipe.load_lora_weights(lora_state_dict)
    pipe.fuse_lora()
    # ------------------------------------------------------------

    # Inverse consistency
    # ------------------------------------------------------------
    stable_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_id,
        unet=unet,
        scheduler=DDIMScheduler.from_pretrained(model_id, subfolder="scheduler"),
        safety_checker=None,
        variant='fp16',
        torch_dtype=torch.float16
    ).to('cuda')
    lora_weight = load_file(forward_checkpoint)

    print(f'Forward CD is loading from {forward_checkpoint}')
    lora_state_dict = get_module_kohya_state_dict(lora_weight, "lora_unet", torch.float32)
    forw_pipe = copy.deepcopy(stable_pipe)
    forw_pipe.load_lora_weights(lora_state_dict)
    forw_pipe.fuse_lora()
    # ------------------------------------------------------------

    return stable_pipe, pipe, forw_pipe


# Load benchmarks (editing or generation)
def load_benchmark(path_to_prompts,
                   path_to_images=None):
    files = pd.read_csv(path_to_prompts)
    if path_to_images is None:
        print(f'Generation benchmark: Loading from {path_to_prompts}')
        prompts = list(files['caption'])
        names = list(files['file_name'])
        return prompts, names
    else:
        print(f'Editing benchmark: Loading prompts, images from {path_to_prompts}, {path_to_images}')
        files = files.reset_index()
        benchmark = []
        for index, row in files.iterrows():
            name = row['file_name']
            img_path = f'{path_to_images}/{name}'
            orig_prompt = row['old_caption']
            edited_prompt = row['edited_caption']
            blended_words = row['blended_words']
            benchmark.append((img_path,
                              {'before': orig_prompt,
                               'after': edited_prompt},
                              blended_words
                              )
                             )
        return benchmark
