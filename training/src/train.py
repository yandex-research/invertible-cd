import copy
from collections import Counter

import torch

from src.lcm import (
    append_dims,
    guidance_scale_embedding,
    predicted_origin,
)


def reverse_train_step(
    args,
    accelerator,
    latents,
    noise,
    prompt_embeds,
    uncond_prompt_embeds,
    encoded_text,
    unet,
    teacher_unet,
    solver,
    w,
    noise_scheduler,
    optimizer,
    lr_scheduler,
    weight_dtype,
    device="cuda",
    uncond_pooled_prompt_embeds=None,
):
    # Sample a random timestep for each image t_n ~ U[0, N - k - 1] without bias.
    index = torch.randint(
        0, args.num_ddim_timesteps, (len(latents),), device=latents.device
    ).long()
    topk = noise_scheduler.config.num_train_timesteps // args.num_ddim_timesteps
    start_timesteps = solver.ddim_timesteps[index]
    timesteps = torch.clamp(start_timesteps - topk, 0, solver.ddim_timesteps[-1])
    assert (start_timesteps > 0).all()

    # Define s for sampled t
    mask = (timesteps[None, :] >= solver.endpoints[:, None]).to(int)
    mask[:-1] = mask[:-1] - mask[1:]
    boundary_timesteps = (mask * solver.endpoints[:, None]).sum(0)

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the reverse diffusion process) [z_{t_{n + k}} in Algorithm 1]
    noisy_model_input = noise_scheduler.add_noise(latents, noise, start_timesteps)

    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)

    # Prepare guidance weights and embeddings
    do_classifier_guidance = (w > 0).any()

    if args.embed_guidance:
        w_embedding = guidance_scale_embedding(
            w.flatten().cpu().float(), embedding_dim=512
        )
        w_embedding = w_embedding.to(device=latents.device, dtype=latents.dtype)
    else:
        w_embedding = None

    noise_pred = unet(
        noisy_model_input,
        start_timesteps,
        timestep_cond=w_embedding,
        encoder_hidden_states=prompt_embeds.float(),
        added_cond_kwargs=encoded_text,
    ).sample

    model_pred = predicted_origin(
        noise_pred,
        start_timesteps,
        boundary_timesteps,
        noisy_model_input,
        noise_scheduler.config.prediction_type,
        alpha_schedule,
        sigma_schedule,
    )

    # Use the ODE solver to predict the kth step in the augmented PF-ODE trajectory after
    # noisy_latents with both the conditioning embedding c and unconditional embedding 0
    # Get teacher model prediction on noisy_latents and conditional embedding
    with torch.no_grad():
        cond_teacher_output = teacher_unet(
            noisy_model_input.to(weight_dtype),
            start_timesteps,
            timestep_cond=w_embedding,
            encoder_hidden_states=prompt_embeds.to(weight_dtype),
            added_cond_kwargs={k: v.to(weight_dtype) for k, v in encoded_text.items()},
        ).sample

        cond_pred_x0 = predicted_origin(
            cond_teacher_output,
            start_timesteps,
            torch.zeros_like(start_timesteps),
            noisy_model_input,
            noise_scheduler.config.prediction_type,
            alpha_schedule,
            sigma_schedule,
        )

        if do_classifier_guidance and not args.embed_guidance:
            # Get teacher model prediction on noisy_latents and unconditional embedding
            if uncond_pooled_prompt_embeds is not None:
                uncond_added_conditions = copy.deepcopy(encoded_text)
                uncond_added_conditions["text_embeds"] = uncond_pooled_prompt_embeds
            else:
                uncond_added_conditions = encoded_text

            uncond_teacher_output = teacher_unet(
                noisy_model_input.to(weight_dtype),
                start_timesteps,
                encoder_hidden_states=uncond_prompt_embeds.to(weight_dtype),
                added_cond_kwargs={
                    k: v.to(weight_dtype) for k, v in uncond_added_conditions.items()
                },
            ).sample
            uncond_pred_x0 = predicted_origin(
                uncond_teacher_output,
                start_timesteps,
                torch.zeros_like(start_timesteps),
                noisy_model_input,
                noise_scheduler.config.prediction_type,
                alpha_schedule,
                sigma_schedule,
            )
            pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
            pred_noise = cond_teacher_output + w * (
                cond_teacher_output - uncond_teacher_output
            )
        else:
            pred_x0 = cond_pred_x0
            pred_noise = cond_teacher_output

        x_prev = solver.ddim_step(pred_x0, pred_noise, index)

    # 20.4.12. Get target LCM prediction on x_prev, w, c, t_n
    with torch.no_grad(), torch.autocast(device, dtype=weight_dtype):
        target_noise_pred = unet(
            x_prev.float(),
            timesteps,
            timestep_cond=w_embedding,
            encoder_hidden_states=prompt_embeds.float(),
            added_cond_kwargs=encoded_text,
        ).sample

        target_pred = predicted_origin(
            target_noise_pred,
            timesteps,
            boundary_timesteps,
            x_prev,
            noise_scheduler.config.prediction_type,
            alpha_schedule,
            sigma_schedule,
        )

        # Apply boundary condition
        boundary_mask = (append_dims(timesteps == boundary_timesteps, x_prev.ndim)).to(
            int
        )
        target_pred = boundary_mask * x_prev + (1 - boundary_mask) * target_pred

    # Calculate loss
    if args.loss_type == "l2":
        loss = F.mse_loss(model_pred.float(), target_pred.float(), reduction="mean")
    elif args.loss_type == "huber":
        loss = torch.mean(
            torch.sqrt(
                (model_pred.float() - target_pred.float()) ** 2 + args.huber_c**2
            ) - args.huber_c
        )

    # Backpropagate on the online student model (`unet`)
    accelerator.backward(loss)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return loss


def forward_train_step(
    args,
    accelerator,
    latents,
    noise,
    prompt_embeds,
    uncond_prompt_embeds,
    encoded_text,
    unet,
    teacher_unet,
    solver,
    w,
    noise_scheduler,
    optimizer,
    lr_scheduler,
    weight_dtype,
    device="cuda",
    uncond_pooled_prompt_embeds=None,
):
    # Sample a random timestep for each image t_n ~ U[0, N - k - 1] without bias.
    index = torch.randint(
        0, args.num_ddim_timesteps - 1, (len(latents),), device=latents.device
    ).long()
    start_timesteps = solver.ddim_timesteps[index]
    max_timestep = solver.ddim_timesteps[args.num_ddim_timesteps - 1]
    assert (start_timesteps >= solver.ddim_timesteps[0]).all() and (
        start_timesteps < max_timestep
    ).all()

    topk = noise_scheduler.config.num_train_timesteps // args.num_ddim_timesteps
    timesteps = torch.clamp(start_timesteps + topk, 0, max_timestep)

    mask = (timesteps[None, :] <= solver.forward_endpoints[:, None]).to(int)
    mask[1:] = mask[1:] - mask[:-1]
    boundary_timesteps = (mask * solver.forward_endpoints[:, None]).sum(0)

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the reverse diffusion process) [z_{t_{n + k}} in Algorithm 1]
    noisy_model_input = noise_scheduler.add_noise(latents, noise, start_timesteps)
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)

    assert (w == 0.0).all()
    do_classifier_guidance = (w > 0).any()
    if args.embed_guidance:
        w_embedding = guidance_scale_embedding(
            w.flatten().cpu().float(), embedding_dim=512
        )
        w_embedding = w_embedding.to(device=latents.device, dtype=latents.dtype)
    else:
        w_embedding = None

    noise_pred = unet(
        noisy_model_input,
        start_timesteps,
        timestep_cond=w_embedding,
        encoder_hidden_states=prompt_embeds.float(),
        added_cond_kwargs=encoded_text,
    ).sample

    model_pred = predicted_origin(
        noise_pred,
        start_timesteps,
        boundary_timesteps,
        noisy_model_input,
        noise_scheduler.config.prediction_type,
        alpha_schedule,
        sigma_schedule,
    )

    # Use the ODE solver to predict the next step in the augmented PF-ODE trajectory after
    # noisy_latents with both the conditioning embedding c and unconditional embedding 0
    # Get teacher model prediction on noisy_latents and conditional embedding
    with torch.no_grad():
        cond_teacher_output = teacher_unet(
            noisy_model_input.to(weight_dtype),
            start_timesteps,
            encoder_hidden_states=prompt_embeds.to(weight_dtype),
            timestep_cond=w_embedding,
            added_cond_kwargs={k: v.to(weight_dtype) for k, v in encoded_text.items()},
        ).sample
        cond_pred_x0 = predicted_origin(
            cond_teacher_output,
            start_timesteps,
            torch.zeros_like(start_timesteps),
            noisy_model_input,
            noise_scheduler.config.prediction_type,
            alpha_schedule,
            sigma_schedule,
        )
        if do_classifier_guidance and not args.embed_guidance:
            # Get teacher model prediction on noisy_latents and unconditional embedding
            if uncond_pooled_prompt_embeds is not None:
                uncond_added_conditions = copy.deepcopy(encoded_text)
                uncond_added_conditions["text_embeds"] = uncond_pooled_prompt_embeds
            else:
                uncond_added_conditions = encoded_text

            uncond_teacher_output = teacher_unet(
                noisy_model_input.to(weight_dtype),
                start_timesteps,
                encoder_hidden_states=uncond_prompt_embeds.to(weight_dtype),
                added_cond_kwargs={
                    k: v.to(weight_dtype) for k, v in uncond_added_conditions.items()
                },
            ).sample
            uncond_pred_x0 = predicted_origin(
                uncond_teacher_output,
                start_timesteps,
                torch.zeros_like(start_timesteps),
                noisy_model_input,
                noise_scheduler.config.prediction_type,
                alpha_schedule,
                sigma_schedule,
            )
            pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
            pred_noise = cond_teacher_output + w * (
                cond_teacher_output - uncond_teacher_output
            )
        else:
            pred_x0 = cond_pred_x0
            pred_noise = cond_teacher_output

        x_next = solver.forward_ddim_step(pred_x0, pred_noise, index)

    # Get target LCM prediction on x_prev, w, c, t_n
    with torch.no_grad(), torch.autocast(device, dtype=weight_dtype):
        target_noise_pred = unet(
            x_next.float(),
            timesteps,
            timestep_cond=w_embedding,
            encoder_hidden_states=prompt_embeds.float(),
            added_cond_kwargs=encoded_text,
        ).sample

        target_pred = predicted_origin(
            target_noise_pred,
            timesteps,
            boundary_timesteps,
            x_next,
            noise_scheduler.config.prediction_type,
            alpha_schedule,
            sigma_schedule,
        )

        # Apply boundary condition
        boundary_mask = (append_dims(timesteps == boundary_timesteps, x_next.ndim)).to(
            int
        )
        target_pred = boundary_mask * x_next + (1 - boundary_mask) * target_pred

    # Calculate loss
    if args.loss_type == "l2":
        loss = F.mse_loss(model_pred.float(), target_pred.float(), reduction="mean")
    elif args.loss_type == "huber":
        loss = torch.mean(
            torch.sqrt(
                (model_pred.float() - target_pred.float()) ** 2 + args.huber_c**2
            ) - args.huber_c
        )

    # Backpropagate on the online student model (`unet`)
    accelerator.backward(loss)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return loss


def forward_preserve_train_step(
    args,
    accelerator,
    latents,
    noise,
    prompt_embeds,
    uncond_prompt_embeds,
    encoded_text,
    forward_unet,
    unet,
    solver,
    w,
    forward_w,
    noise_scheduler,
    optimizer,
    lr_scheduler,
    weight_dtype,
    device="cuda",
):
    assert len(solver.forward_endpoints) <= len(solver.endpoints)
    num_reverse_steps = len(solver.endpoints) // len(solver.forward_endpoints)

    # Sample time ranges
    endpoint_index = torch.randint(
        0, len(solver.forward_endpoints), (len(latents),), device=latents.device
    ).long()
    start_timesteps = solver.forward_endpoints[endpoint_index]
    left_end_timesteps = solver.endpoints[::num_reverse_steps][endpoint_index]
    left_end_timesteps[left_end_timesteps == 0] = args.start_forward_timestep

    # Steps to sample using the reverse unet
    reverse_timesteps = solver.endpoints.reshape(-1, num_reverse_steps)[endpoint_index]
    reverse_timesteps[reverse_timesteps == 0] = args.start_forward_timestep
    assert (reverse_timesteps[:, 0] == left_end_timesteps).all()

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the reverse diffusion process) [z_{t_{n + k}} in Algorithm 1]
    start_model_input = noise_scheduler.add_noise(
        latents, noise, start_timesteps
    )  # Target
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)

    if args.embed_guidance:
        reverse_w = torch.zeros_like(w)  # Preserve losses work only for CFG=0
        reverse_w_embedding = guidance_scale_embedding(
            reverse_w.flatten().cpu().float(), embedding_dim=512
        )
        reverse_w_embedding = reverse_w_embedding.to(
            device=latents.device, dtype=latents.dtype
        )

        assert (forward_w == 0.0).all()
        forward_w_embedding = guidance_scale_embedding(
            forward_w.flatten().cpu().float(), embedding_dim=512
        )
        forward_w_embedding = forward_w_embedding.to(
            device=latents.device, dtype=latents.dtype
        )
    else:
        reverse_w_embedding = None
        forward_w_embedding = None

    # Init reverse inputs
    current_timesteps = start_timesteps
    reverse_input = start_model_input

    # reverse sampling
    for i in range(num_reverse_steps):
        with torch.no_grad(), torch.autocast(device, dtype=weight_dtype):
            noise_pred = unet(
                reverse_input,
                current_timesteps,
                timestep_cond=reverse_w_embedding,
                encoder_hidden_states=prompt_embeds.float(),
                added_cond_kwargs=encoded_text,
            ).sample

        # Make a reverse step
        next_timesteps = reverse_timesteps[:, num_reverse_steps - i - 1]
        reverse_output = predicted_origin(
            noise_pred,
            current_timesteps,
            next_timesteps,
            reverse_input,
            noise_scheduler.config.prediction_type,
            alpha_schedule,
            sigma_schedule,
        )
        # Next step init
        current_timesteps = next_timesteps
        reverse_input = reverse_output

    # Get target LCM prediction on x_prev, w, c, t_n
    target_noise_pred = forward_unet(
        reverse_output,
        left_end_timesteps,
        timestep_cond=forward_w_embedding,
        encoder_hidden_states=prompt_embeds.float(),
        added_cond_kwargs=encoded_text,
    ).sample

    # Final prediction
    start_model_pred = predicted_origin(
        target_noise_pred,
        left_end_timesteps,
        start_timesteps,
        reverse_output,
        noise_scheduler.config.prediction_type,
        alpha_schedule,
        sigma_schedule,
    )

    logs = Counter()
    # Calculate loss
    if args.loss_type == "l2":
        loss = F.mse_loss(
            start_model_pred.float(), start_model_input.float(), reduction="mean"
        )
    elif args.loss_type == "huber":
        losses = (
            torch.sqrt(
                (start_model_pred.float() - start_model_input.float()) ** 2
                + args.huber_c**2
            ) - args.huber_c
        )
        reduce_dims = tuple(range(len(start_model_pred.shape)))[1:]
        losses = losses.mean(reduce_dims)

        for start_timestep, end_timestep, loss in zip(
            start_timesteps, left_end_timesteps, losses
        ):
            logs[
                f"forward_preserve_loss_{start_timestep}_{end_timestep}"
            ] += loss.detach().item() / len(latents)
        loss = losses.mean()

    # Backpropagate on the online student model (`unet`)
    accelerator.backward(loss * args.forward_preserve_loss_coef)
    logs[f"forward_preserve_loss"] = loss.detach().item()

    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(forward_unet.parameters(), args.max_grad_norm)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return logs


def reverse_preserve_train_step(
    args,
    accelerator,
    latents,
    noise,
    prompt_embeds,
    uncond_prompt_embeds,
    encoded_text,
    forward_unet,
    unet,
    solver,
    w,
    forward_w,
    noise_scheduler,
    optimizer,
    lr_scheduler,
    weight_dtype,
    device="cuda",
):
    assert len(solver.forward_endpoints) <= len(solver.endpoints)
    num_reverse_steps = len(solver.endpoints) // len(solver.forward_endpoints)

    # Sample time ranges
    endpoint_index = torch.randint(
        0, len(solver.forward_endpoints), (len(latents),), device=latents.device
    ).long()
    start_timesteps = solver.endpoints[::num_reverse_steps][endpoint_index]
    start_timesteps[start_timesteps == 0] = args.start_forward_timestep
    end_timesteps = solver.forward_endpoints[endpoint_index]

    # Steps to sample using the reverse unet
    reverse_timesteps = solver.endpoints.reshape(-1, num_reverse_steps)[endpoint_index]
    reverse_timesteps[reverse_timesteps == 0] = args.start_forward_timestep
    assert (reverse_timesteps[:, 0] == start_timesteps).all()

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the reverse diffusion process) [z_{t_{n + k}} in Algorithm 1]
    start_model_input = noise_scheduler.add_noise(
        latents, noise, start_timesteps
    )  # Target
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)

    if args.embed_guidance:
        reverse_w = torch.zeros_like(w)  # Preserve losses work only for CFG=0
        reverse_w_embedding = guidance_scale_embedding(
            reverse_w.flatten().cpu().float(), embedding_dim=512
        )
        reverse_w_embedding = reverse_w_embedding.to(
            device=latents.device, dtype=latents.dtype
        )

        assert (forward_w == 0.0).all()
        forward_w_embedding = guidance_scale_embedding(
            forward_w.flatten().cpu().float(), embedding_dim=512
        )
        forward_w_embedding = forward_w_embedding.to(
            device=latents.device, dtype=latents.dtype
        )
    else:
        reverse_w_embedding = None
        forward_w_embedding = None

    with torch.no_grad(), torch.autocast(device, dtype=weight_dtype):
        noise_pred = forward_unet(
            start_model_input,
            start_timesteps,
            timestep_cond=forward_w_embedding,
            encoder_hidden_states=prompt_embeds.float(),
            added_cond_kwargs=encoded_text,
        ).sample

        forward_output = predicted_origin(
            noise_pred,
            start_timesteps,
            end_timesteps,
            start_model_input,
            noise_scheduler.config.prediction_type,
            alpha_schedule,
            sigma_schedule,
        )

    # Init reverse inputs
    reverse_input = forward_output
    current_timesteps = end_timesteps

    # reverse sampling
    for i in range(num_reverse_steps):
        noise_pred = unet(
            reverse_input,
            current_timesteps,
            timestep_cond=reverse_w_embedding,
            encoder_hidden_states=prompt_embeds.float(),
            added_cond_kwargs=encoded_text,
        ).sample

        # Make a reverse step
        next_timesteps = reverse_timesteps[:, num_reverse_steps - i - 1]
        reverse_output = predicted_origin(
            noise_pred,
            current_timesteps,
            next_timesteps,
            reverse_input,
            noise_scheduler.config.prediction_type,
            alpha_schedule,
            sigma_schedule,
        )
        # Next step init
        current_timesteps = next_timesteps
        reverse_input = reverse_output

    logs = Counter()
    # 20.4.13. Calculate loss
    if args.loss_type == "l2":
        loss = F.mse_loss(
            reverse_output.float(), start_model_input.float(), reduction="mean"
        )
    elif args.loss_type == "huber":
        losses = (
            torch.sqrt(
                (reverse_output.float() - start_model_input.float()) ** 2
                + args.huber_c**2
            ) - args.huber_c
        )
        reduce_dims = tuple(range(len(start_model_input.shape)))[1:]
        losses = losses.mean(reduce_dims)

        for start_timestep, end_timestep, loss in zip(
            start_timesteps, end_timesteps, losses
        ):
            logs[
                f"reverse_preserve_loss_{start_timestep}_{end_timestep}"
            ] += loss.detach().item() / len(latents)
        loss = losses.mean()

    # 20.4.14. Backpropagate on the online student model (`unet`)
    accelerator.backward(loss * args.reverse_preserve_loss_coef)
    logs[f"reverse_preserve_loss"] = loss.detach().item()

    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return logs
