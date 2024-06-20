import torch.nn.functional as nnf
import torch

from tqdm import tqdm
from torch.optim.adam import Adam

from utils.generation import load_512
from utils.p2p import register_attention_control


def null_optimization(solver,
                      latents,
                      guidance_scale,
                      num_inner_steps,
                      epsilon):
    uncond_embeddings, cond_embeddings = solver.context.chunk(2)
    uncond_embeddings_list = []
    latent_cur = latents[-1]
    bar = tqdm(total=num_inner_steps * solver.n_steps)
    for i in range(solver.n_steps):
        uncond_embeddings = uncond_embeddings.clone().detach()
        uncond_embeddings.requires_grad = True
        optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
        latent_prev = latents[len(latents) - i - 2]
        t = solver.model.scheduler.timesteps[i]
        with torch.no_grad():
            noise_pred_cond = solver.get_noise_pred_single(latent_cur, t, cond_embeddings)
        for j in range(num_inner_steps):
            noise_pred_uncond = solver.get_noise_pred_single(latent_cur, t, uncond_embeddings)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            latents_prev_rec = solver.prev_step(noise_pred, t, latent_cur)
            loss = nnf.mse_loss(latents_prev_rec, latent_prev)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_item = loss.item()
            bar.update()
            if loss_item < epsilon + i * 2e-5:
                break
        for j in range(j + 1, num_inner_steps):
            bar.update()
        uncond_embeddings_list.append(uncond_embeddings[:1].detach())
        with torch.no_grad():
            context = torch.cat([uncond_embeddings, cond_embeddings])
            noise_pred = solver.get_noise_pred(solver.model, latent_cur, t, guidance_scale, context)
            latent_cur = solver.prev_step(noise_pred, t, latent_cur)
    bar.close()
    return uncond_embeddings_list


def invert(solver, 
           stop_step,
           is_cons_inversion=False,
           inv_guidance_scale=1, 
           nti_guidance_scale=8,
           dynamic_guidance=False,
           tau1=0.4,
           tau2=0.6,
           w_embed_dim=0,
           image_path=None, 
           prompt='',
           offsets=(0, 0, 0, 0),
           do_nti=False,
           do_npi=False,
           num_inner_steps=10, 
           early_stop_epsilon=1e-5,
           seed=0,
           ):
    solver.init_prompt(prompt)
    uncond_embeddings, cond_embeddings = solver.context.chunk(2)
    register_attention_control(solver.model, None)
    if isinstance(image_path, list):
        image_gt = [load_512(path, *offsets) for path in image_path]
    else:
        image_gt = load_512(image_path, *offsets)
    
    if is_cons_inversion:
        image_rec, ddim_latents = solver.cons_inversion(image_gt,
                                                        w_embed_dim=w_embed_dim,
                                                        guidance_scale=inv_guidance_scale,
                                                        seed=seed,)
    else:  
        image_rec, ddim_latents = solver.ddim_inversion(image_gt, 
                                                        n_steps=stop_step,
                                                        guidance_scale=inv_guidance_scale,
                                                        dynamic_guidance=dynamic_guidance,
                                                        tau1=tau1, tau2=tau2, 
                                                        w_embed_dim=w_embed_dim)
    if do_nti:
        print("Null-text optimization...")
        uncond_embeddings = null_optimization(solver,
                                              ddim_latents,
                                              nti_guidance_scale,
                                              num_inner_steps,
                                              early_stop_epsilon)
    elif do_npi:
        uncond_embeddings = [cond_embeddings] * solver.n_steps
    else:
        uncond_embeddings = None
    return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings