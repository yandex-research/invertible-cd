import torch
import numpy as np
import argparse
import copy
import sys
import json
import pickle
from nltk.corpus import stopwords
import os

from safetensors.torch import load_file
from peft import LoraConfig
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel,DDPMScheduler
from tqdm import tqdm

from utils import p2p, generation, inversion, metrics
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

def find_difference(word1, word2):
    splitted_w1 = word1.split(' ')
    splitted_w2 = word2.split(' ')
    for i,j in zip(splitted_w1, splitted_w2):
        if i != j:
            return i, j

        
def find_difference2(word1, word2):
    splitted_w1 = word1.split(' ')
    splitted_w2 = word2.split(' ')
    out = []
    for i in splitted_w2:
        if i not in splitted_w1:
            out.append(i)
            
    return out

def n_differences(word1, word2):
    splitted_w1 = word1.split(' ')
    splitted_w2 = word2.split(' ')
    diff = 0
    for i,j in zip(splitted_w1, splitted_w2):
        if i != j:
            diff += 1
    return diff
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
        help="Path to forward CM",
    )    
    parser.add_argument(
        "--forward_checkpoint",
        type=str,
        default=None,
        help="Path to inverse CM",
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
    parser.add_argument(
        "--path_to_images",
        type=str,
        default=None,
        required=False,
        help="Path to images for editing benchmarking only",
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
        required=False,
        help="number of steps for reverse CD",
    ) 
    parser.add_argument(
        "--num_forward_cons_steps",
        type=int,
        default=3,
        required=False,
        help="number of steps for forward CD",
    ) 
    parser.add_argument(
        "--max_forward_timestep_idx",
        type=int,
        default=49,
        help="the last timestep for forward CD for encode",
    ) 
    parser.add_argument(
        "--start_timestep",
        type=int,
        default=19,
        help="starting timestep for noising real images",
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
    
    # Inversion settings
    ################################
    parser.add_argument(
        "--use_cons_inversion",
        type=str2bool,
        default='True',
        required=True,
        help='whether to do inversion with CM'
    )
    parser.add_argument(
        "--nti_guidance_scale",
        type=float,
        default=8.0,
        help="guidance scale for inversion with NTI",
    )
    parser.add_argument(
        "--use_npi",
        type=str2bool,
        default='False',
        help='whether to use negative prompt inversion'
    )
    parser.add_argument(
        "--use_nti",
        type=str2bool,
        default='False',
        help='whether to use null text inversion'
    )
    ################################
    
    # Editing settings
    ################################
    parser.add_argument(
        "--use_cons_editing",
        type=str2bool,
        default='True',
        required=True,
        help='whether to do editing with CM'
    )
    parser.add_argument(
        "--dynamic_guidance",
        type=str2bool,
        default='False',
        required=False,
        help='whether to use dynamic guidance for editing'
    )
    parser.add_argument(
        "--tau1",
        type=float,
        default=1.0,
        required=False,
        help="first hyperparameter for dynamic guidance",
    )
    parser.add_argument(
        "--tau2",
        type=float,
        default=1.0,
        required=False,
        help="second hyperparameter for dynamic guidance",
    )
    parser.add_argument(
        "--cross_replace_steps",
        type=float,
        default=0.4
    )
    parser.add_argument(
        "--self_replace_steps",
        type=float,
        default=0.4
    )
    parser.add_argument(
        "--amplify_factor",
        type=float,
        default=3
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=8.0,
        required=True,
        help="guidance scale for editing",
    )
    ################################
    
    # Others
    ################################
    parser.add_argument(
        "--is_replacement",
        type=str2bool,
        default='True',
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=999999,
    )
    parser.add_argument(
        "--saving_dir",
        type=str,
        required=True,
        default='results',
    )
    parser.add_argument(
        "--path_to_uncond_embeddings",
        type=str,
        default='',
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
def main(args):
    
    # Models loading
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
    editing_benchmark = load_benchmark(args.path_to_prompts,
                                       args.path_to_images)

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
    
    # EDITING PART
    print('Running editing...')
    eval_collection = {'orig_prompt': [], 'orig_image': [], 'edited_prompt': [], 'edited_image': []}
    
    if len(args.path_to_uncond_embeddings) > 0:
        with open(args.path_to_uncond_embeddings, 'rb') as handle:
            print(f'Loading uncond embeddings from {args.path_to_uncond_embeddings}')
            uncond_embeddings_collection = pickle.load(handle)

    for j, (image_path, prompts_dict, blended_words) in enumerate(tqdm(editing_benchmark)):

        if args.is_replacement:
            if n_differences(prompts_dict['before'], prompts_dict['after']) != 1:
                continue
            if len(prompts_dict['before'].split(' ')) != len(prompts_dict['after'].split(' ')):
                continue

        prompt = [prompts_dict['before']]
        
        if len(args.path_to_uncond_embeddings) > 0:
            if prompts_dict['before'] in uncond_embeddings_collection.keys():
                args.use_nti = False
            else:
                args.use_nti = True
        
        (image_gt, image_rec), latent, uncond_embeddings = inversion.invert(
                                                                   # Playing params
                                                                   is_cons_inversion=args.use_cons_inversion,
                                                                   do_npi=args.use_npi,
                                                                   do_nti=args.use_nti,
                                                                   stop_step=50, # from [0, NUM_DDIM_STEPS]
                                                                   w_embed_dim=args.w_embed_dim,
                                                                   inv_guidance_scale=0.0,
        
                                                                   nti_guidance_scale=args.nti_guidance_scale,
                                                                   dynamic_guidance=False,
                                                                   tau1=0.0,
                                                                   tau2=0.0,
    
                                                                   # Fixed params
                                                                   solver=generator,
                                                                   image_path=image_path, 
                                                                   prompt=prompt,
                                                                   offsets=(0,0,200,0),
                                                                   num_inner_steps=10, 
                                                                   early_stop_epsilon=1e-5,
                                                                   seed=args.seed)
        
        if len(args.path_to_uncond_embeddings) > 0:
            if not args.use_nti:
                uncond_embeddings = uncond_embeddings_collection[prompts_dict['before']]
                uncond_embeddings = [item.to(args.device) for item in uncond_embeddings]
            else:
                uncond_embeddings_collection[prompts_dict['before']] = [item.to('cpu') for item in uncond_embeddings]
        
        if args.use_cons_editing:
            p2p.NUM_DDIM_STEPS = args.num_reverse_cons_steps
            model = reverse_cons_model
        else:
            model = ldm_stable
        
        prompts = [prompts_dict['before'], prompts_dict['after']]
        cross_replace_steps = {'default_': args.cross_replace_steps,} 
        self_replace_steps = args.self_replace_steps
        blend_word = None
        eq_params = None
        if args.is_replacement:
            is_replacement = True
            w = find_difference(prompts_dict['before'], prompts_dict['after'])
            w1, w2 = w[0], w[1]
            blend_word = (((w1,), (w2,)))
            eq_params = {"words": (w2,), "values": (args.amplify_factor,)} 
        else:
            is_replacement = False
            if blended_words != " ":
                blended_words = blended_words.split(" ")
                blend_word = (((blended_words[0],), (blended_words[1],)))
            w = find_difference2(prompts_dict['before'], prompts_dict['after'])
            w = [word for word in w if word not in stopwords.words('english')]
            if len(prompts_dict['before'].split(' ')) == len(prompts_dict['after'].split(' ')):
                is_replacement = True
            b_w = tuple([i for i in w])
            a_f = tuple([args.amplify_factor for _ in range(len(w))])
            eq_params = {"words": b_w, "values": a_f}

        controller = p2p.make_controller(prompts, 
                                         is_replacement,
                                         cross_replace_steps,
                                         self_replace_steps, 
                                         blend_word, 
                                         eq_params)
        image, _ = generation.runner(
                                 # Playing params
                                 model=model, # ldm_stable or forw_cons_model
                                 is_cons_forward=args.use_cons_editing,
                                
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
                                 generator=None,
                                 latent=latent,
                                 uncond_embeddings=uncond_embeddings,
                                 return_type='image')
        
        pil_img_orig = generation.to_pil_images(image_gt)
        pil_img_edited = generation.to_pil_images(image[1, :, :, :])
        eval_collection['orig_prompt'].append(prompts_dict['before'])
        eval_collection['orig_image'].append(pil_img_orig)
        eval_collection['edited_prompt'].append(prompts_dict['after'])
        eval_collection['edited_image'].append(pil_img_edited)
        
    # VALIDATION PART        
    preserve_clip_score = metrics.calc_clip_score_images_images(eval_collection['orig_image'], 
                                                                eval_collection['edited_image'], 
                                                                device=args.device, batch_size=16)
    preserve_dinov2 = metrics.calc_dinov2_images_images(eval_collection['orig_image'], 
                                                        eval_collection['edited_image'], 
                                                        device=args.device, batch_size=16)
    editing_clip_score = metrics.calc_clip_score_images_prompts(eval_collection['edited_image'], 
                                                                eval_collection['edited_prompt'], 
                                                                device=args.device, batch_size=16)
    editing_imagereward = metrics.calc_ir(eval_collection['edited_image'], 
                                          eval_collection['edited_prompt'], 
                                          device=args.device, batch_size=16)
    results = {'preservation_clip_score': str(list(np.array(preserve_clip_score))), 
               'preservation_dinov2': str(list(np.array(preserve_dinov2))),
               'editing_clip_score': str(list(np.array(editing_clip_score))),
               'editing_imagereward': str(editing_imagereward)}
    
    # SAVING PART
    outdir = args.saving_dir
    os.makedirs(outdir, exist_ok=True)
    with open(f'{outdir}/editing_metrics_values.json', "w") as fp:
        json.dump(results , fp)
        
    outdir_images = f'{args.saving_dir}/edited_images'
    os.makedirs(outdir_images, exist_ok=True)
    for j, image in enumerate(eval_collection['edited_image']):
        image.save(f'{outdir_images}/{j}.jpg')
# -------------------------------------------------------------------------------------


if __name__ == "__main__":
    args = parse_args()
    main(args)
