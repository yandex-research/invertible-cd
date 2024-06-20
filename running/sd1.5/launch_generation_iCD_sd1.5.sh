####################################################################
# Generation with Consistency Models
####################################################################

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 sd1.5/generate.py               \
                                                   --reverse_checkpoint <your_path>             \
                                                   --forward_checkpoint <your_path>             \
                                                   --teacher_checkpoint <your_path>             \
                                                   --num_reverse_cons_steps 4                   \
                                                   --reverse_timesteps 259 519 779 999          \
                                                   --num_forward_cons_steps 4                   \
                                                   --forward_timesteps 19 259 519 779           \
                                                   --path_to_prompts benchmarks/instructions/generation_coco.csv        \
                                                   --guidance_scale 7.0                         \
                                                   --dynamic_guidance False                     \
                                                   --tau1 1.0                                   \
                                                   --tau2 1.0                                   \
                                                   --batch_per_gpu 8                            \
                                                   --max_cnt 128                                \
                                                   --saving_dir results_generation_cons         \
                                                   --model_id_DM runwayml/stable-diffusion-v1-5 \
                                                   --max_forward_timestep_idx 49                \
                                                   --start_timestep 19                          \
                                                   --inv_guidance_scale 0                       \
                                                   --lora_rank 64                               \
                                                   --w_embed_dim 512                            \
                                                   --num_ddim_steps 50                          \
                                                   --use_cons_generation True                   \
                                                   --use_cons_inversion True                    \
                                                   --device cuda                                \
                                                   --dtype fp16                                 \
                                                   --seed 453645634
