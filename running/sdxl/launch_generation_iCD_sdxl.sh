####################################################################
# Generation with invertible Consistency Distillation on SDXL
####################################################################

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 sdxl/generate.py \
                                                   --model_id stabilityai/stable-diffusion-xl-base-1.0  \
                                                   --reverse_checkpoint <your_path>     \
                                                   --forward_checkpoint <your_path>     \
                                                   --teacher_checkpoint <your_path>     \
                                                   --n_steps 4                          \
                                                   --reverse_timesteps 249 499 699 999  \
                                                   --path_to_prompts benchmarks/instructions/generation_parti-prompts.csv  \
                                                   --guidance_scale 7.0                 \
                                                   --use_dynamic_guidance False         \
                                                   --tau1 1.0                           \
                                                   --tau2 1.0                           \
                                                   --batch_per_gpu 8                    \
                                                   --max_cnt 128                        \
                                                   --saving_dir results_generation_icd_sdxl    \
