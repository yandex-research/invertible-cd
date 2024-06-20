import argparse
import torch
import sys
import pickle
import os
import torch.distributed as dist
import numpy as np
import pandas as pd

from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel,DDPMScheduler
from tqdm import tqdm
from PIL import Image

from utils import p2p, generation, inversion, metrics, dist_utils
from utils.loading import load_models, load_benchmark

# Utils
# -------------------------------------------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def prepare_val_prompts(all_text, bs=20, max_cnt=5000):
    all_text = all_text[:max_cnt]

    num_batches = ((len(all_text) - 1) // (bs * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = np.array_split(np.array(all_text), num_batches)
    rank_batches = all_batches[dist.get_rank():: dist.get_world_size()]

    index_list = np.arange(len(all_text))
    all_batches_index = np.array_split(index_list, num_batches)
    rank_batches_index = all_batches_index[dist.get_rank():: dist.get_world_size()]
    return rank_batches, rank_batches_index, all_text        
# -------------------------------------------------------------------------------------


# Arguments parser
# -------------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    
    # Loading settings
    ################################
    parser.add_argument(
        "--model_id_DM",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained DM",
    )
    parser.add_argument(
        "--reverse_checkpoint",
        type=str,
        default=None,
        help="Path to reverse CM",
    )    
    parser.add_argument(
        "--forward_checkpoint",
        type=str,
        default=None,
        help="Path to forward CM",
    )   
    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        default=None,
        help="Path to teacher DM with w embedding",
    ) 
    parser.add_argument(
        "--path_to_prompts",
        type=str,
        default=None,
        required=True,
        help="Path to prompts for benchmarking",
    ) 
    ################################
    
    # Models settings
    ################################
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="rank of lora weights",
    ) 
    parser.add_argument(
        "--w_embed_dim",
        type=int,
        default=0,
        help="dimension of guidance embedding",
    ) 
    parser.add_argument(
        "--num_ddim_steps",
        type=int,
        default=50,
    ) 
    parser.add_argument(
        "--num_reverse_cons_steps",
        type=int,
        default=4,
        required=True,
        help="number of steps for forward CM",
    ) 
    parser.add_argument(
        "--num_forward_cons_steps",
        type=int,
        default=3,
        required=True,
        help="number of steps for inverse CM",
    ) 
    parser.add_argument(
        "--max_forward_timestep_idx",
        type=int,
        default=49,
        help="the last timestep for inverse CM for encode",
    ) 
    parser.add_argument(
        "--start_timestep",
        type=int,
        default=19,
        help="starting timestep for noising real images",
    )
    ################################
    
    # Generation settings
    ################################
    parser.add_argument(
        "--use_cons_generation",
        type=str2bool,
        default='True',
        required=True,
        help='whether to do generation with CM'
    )
    parser.add_argument(
        "--dynamic_guidance",
        type=str2bool,
        default='True',
        required=True,
        help='whether to use dynamic guidance for editing'
    )
    parser.add_argument(
        "--tau1",
        type=float,
        default=0.8,
        required=True,
        help="first hyperparameter for dynamic guidance",
    )
    parser.add_argument(
        "--tau2",
        type=float,
        default=0.8,
        required=True,
        help="second hyperparameter for dynamic guidance",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=8.0,
        required=True,
        help="guidance scale for editing",
    )
    parser.add_argument(
        "--reverse_timesteps",
        default='',
        nargs='+',
        required=False,
    )
    parser.add_argument(
        "--forward_timesteps",
        '--list',
        default='',
        nargs='+',
        required=False,
    )
    ################################
    
    # Others
    ################################
    parser.add_argument(
        "--use_cons_inversion",
        type=str2bool,
        default='False',
        required=True,
        help='whether to do generation with CM'
    )
    parser.add_argument(
        "--inv_guidance_scale",
        type=int,
        default=1.0,
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
    )
    parser.add_argument(
        "--batch_per_gpu",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--max_cnt",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--saving_dir",
        type=str,
        required=True,
        default='results',
    )
    parser.add_argument(
        "--path_to_images",
        type=str,
        required=False,
        default='',
    )
    parser.add_argument(
        "--path_to_fid_reference",
        type=str,
        default='files/fid_stats_mscoco512_val.npz',
    )
    parser.add_argument(
        "--path_to_inception",
        type=str,
        default='files/pt_inception-2015-12-05-6726825d.pth',
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default='fp32',
    )
    ################################
    
    args = parser.parse_args()

    return args
# -------------------------------------------------------------------------------------

# Running 
# -------------------------------------------------------------------------------------
@torch.no_grad()
def main(args):
    
    # Models loading
    dist_utils.init()
    ldm_stable, reverse_cons_model, forward_cons_model = load_models(
        model_id=args.model_id_DM,
        device=args.device,
        reverse_checkpoint=args.reverse_checkpoint,
        forward_checkpoint=args.forward_checkpoint,
        r=args.lora_rank,
        w_embed_dim=args.w_embed_dim,
        teacher_checkpoint=args.teacher_checkpoint,
        dtype=args.dtype)

    tokenizer = ldm_stable.tokenizer
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.model_id_DM, subfolder="scheduler", 
    )
    
    # Benchmark loading
    generation_benchmark, names = load_benchmark(args.path_to_prompts)
    rank_batches, rank_batches_index, all_prompts = prepare_val_prompts(generation_benchmark,
                                                                        bs=args.batch_per_gpu,
                                                                        max_cnt=args.max_cnt)
    if len(args.reverse_timesteps) > 0 and len(args.forward_timesteps) > 0:
        reverse_timesteps = [int(i) for i in args.reverse_timesteps]
        forward_timesteps = [int(i) for i in args.forward_timesteps]
    else:
        reverse_timesteps = None
        forward_timesteps = None

    # Generator configuration
    generator = generation.Generator(
                            model=ldm_stable, 
                            noise_scheduler=noise_scheduler,
                            n_steps=args.num_ddim_steps,
                            forward_cons_model=forward_cons_model,
                            reverse_cons_model=reverse_cons_model,
                            num_endpoints=args.num_reverse_cons_steps,
                            num_forward_endpoints=args.num_forward_cons_steps,
                            max_forward_timestep_index=args.max_forward_timestep_idx,
                            reverse_timesteps=reverse_timesteps,
                            forward_timesteps=forward_timesteps,
                            start_timestep=args.start_timestep)
    p2p.NUM_DDIM_STEPS = args.num_ddim_steps
    p2p.tokenizer = tokenizer
    p2p.device = args.device
    
    # GENERATION PART
    local_images, local_text_idxs = [], []
    if len(args.path_to_images) > 0:
        local_images_orig = []
    generator_seed = torch.Generator().manual_seed(args.seed)
    for cnt, mini_batch in enumerate(tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0))):
        if args.use_cons_generation:
            p2p.NUM_DDIM_STEPS = args.num_reverse_cons_steps
            model = reverse_cons_model
        else:
            model = ldm_stable    
            
        controller = p2p.AttentionStore()
        prompts = list(mini_batch)
        
        if len(args.path_to_images) > 0:
            image_path = [f'{args.path_to_images}/{names[c]}' for c in rank_batches_index[cnt]]
            (image_gt, image_rec), latent, uncond_embeddings = inversion.invert(
                                                                   # Playing params
                                                                   is_cons_inversion=args.use_cons_inversion,
                                                                   do_npi=False,
                                                                   do_nti=False,
                                                                   w_embed_dim=0,
                                                                   stop_step=50, # from [0, NUM_DDIM_STEPS]
        
                                                                   nti_guidance_scale=8.0,
                                                                   inv_guidance_scale=args.inv_guidance_scale,
                                                                   dynamic_guidance=False,
                                                                   tau1=0.0,
                                                                   tau2=0.0,
    
                                                                   # Fixed params
                                                                   solver=generator,
                                                                   image_path=image_path, 
                                                                   prompt=prompts,
                                                                   offsets=(0,0,200,0),
                                                                   num_inner_steps=10, 
                                                                   early_stop_epsilon=1e-5,
                                                                   seed=args.seed)
        else:
            latent = None
    
        image, _ = generation.runner(
                                 # Playing params
                                 model=model, # ldm_stable or forw_cons_model
                                 is_cons_forward=args.use_cons_generation,
                                 w_embed_dim=args.w_embed_dim,
                                 guidance_scale=args.guidance_scale,
                                 dynamic_guidance=args.dynamic_guidance,
                                 tau1=args.tau1,
                                 tau2=args.tau2,
                                 start_time=50,
    
                                 # Fixed params
                                 solver=generator,
                                 prompt=prompts,
                                 controller=controller,
                                 num_inference_steps=50,
                                 generator=generator_seed,
                                 latent=latent,
                                 return_type='image')
        
        for text_idx, global_idx in enumerate(rank_batches_index[cnt]):
            img_tensor = torch.tensor(np.array(image[text_idx]))
            local_images.append(img_tensor)
            local_text_idxs.append(global_idx)
            if len(args.path_to_images) > 0:
                img_tensor_orig = torch.tensor(np.array(image_gt[text_idx]))
                local_images_orig.append(img_tensor_orig)
        
    local_images = torch.stack(local_images).cuda()
    local_text_idxs = torch.tensor(local_text_idxs).cuda()
    
    gathered_images = [torch.zeros_like(local_images) for _ in range(dist.get_world_size())]
    gathered_text_idxs = [torch.zeros_like(local_text_idxs) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_images, local_images)  # gather not supported with NCCL
    dist.all_gather(gathered_text_idxs, local_text_idxs)
    
    if len(args.path_to_images) > 0:
        local_images_orig = torch.stack(local_images_orig).cuda()
        gathered_images_orig = [torch.zeros_like(local_images_orig) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_images_orig, local_images_orig)  # gather not supported with NCCL
    
    images, prompts = [], []
    if dist.get_rank() == 0:
        gathered_images = np.concatenate(
                [images.cpu().numpy() for images in gathered_images], axis=0
            )
        gathered_text_idxs = np.concatenate(
                [text_idxs.cpu().numpy() for text_idxs in gathered_text_idxs], axis=0
            )
        for image, global_idx in zip(gathered_images, gathered_text_idxs):
            item = np.array(image)
            img = Image.fromarray(item.astype('uint8'))
            images.append(img)
            prompts.append(all_prompts[global_idx])
            
        outdir_images = f'{args.saving_dir}/generated_images'
        os.makedirs(outdir_images, exist_ok=True)
        for j, image in enumerate(images):
            image.save(f'{outdir_images}/{j}.jpg')
        # VALIDATION PART
        if len(args.path_to_images) == 0:
            clip_score = torch.mean(metrics.calc_clip_score_images_prompts(
                                       images,
                                       prompts,
                                       args.device,
                                       args.batch_per_gpu))
            ir = np.mean(metrics.calc_ir(
                                 images,
                                 prompts,
                                 args.device,
                                 args.batch_per_gpu))
            results = {'clip_score': clip_score,
                       'ir': ir}
            print(results)
            
            outdir = args.saving_dir
            os.makedirs(outdir, exist_ok=True)
            with open(f'{outdir}/generation_metrics_values.pickle', 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

            dataframe=pd.DataFrame(data=prompts,columns=['prompts'])
            dataframe.to_csv(f'{outdir}/prompts.csv')
        else:
            gathered_images_orig = np.concatenate(
                [images.cpu().numpy() for images in gathered_images_orig], axis=0
            )
            images = []
            for image in gathered_images_orig:
                item = np.array(image)
                img = Image.fromarray(item.astype('uint8'))
                images.append(img)
            outdir_images = f'{args.saving_dir}/real_images'
            os.makedirs(outdir_images, exist_ok=True)
            for j, image in enumerate(images):
                image.save(f'{outdir_images}/{j}.jpg')
# -------------------------------------------------------------------------------------


if __name__ == "__main__":
    args = parse_args()
    main(args)
