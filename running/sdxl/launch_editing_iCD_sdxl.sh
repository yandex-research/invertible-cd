####################################################################
# Editing with invertible Consistency Distillation on SDXL
####################################################################

CUDA_VISIBLE_DEVICES=0 python3.9 sdxl/edit.py \
                                                   --model_id stabilityai/stable-diffusion-xl-base-1.0  \
                                                   --reverse_checkpoint <your_path>      \
                                                   --forward_checkpoint <your_path>      \
                                                   --teacher_checkpoint <your_path>      \
                                                   --n_steps 4                           \
                                                   --reverse_timesteps 249 499 699 999   \
                                                   --forward_timesteps 19 249 499 699    \
                                                   --path_to_prompts benchmarks/instructions/editing_pie_bench_140.csv        \
                                                   --path_to_images benchmarks/images/pie_bench_140_images         \
                                                   --guidance_scale 19.0                 \
                                                   --use_dynamic_guidance True           \
                                                   --tau1 0.7                            \
                                                   --tau2 0.7                            \
                                                   --saving_dir results_editing_icd_sdxl \
