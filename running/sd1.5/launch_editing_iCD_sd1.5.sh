####################################################################
# Editing with CD for different hyperparameters
####################################################################

for crs_srs in "0.3 0.6"
do
    set -- $crs_srs
    echo "CD: cross_replace_steps: $1, self_replace_steps $2"
    CUDA_VISIBLE_DEVICES=0 python3.9 sd1.5/edit.py --model_id_DM runwayml/stable-diffusion-v1-5         \
                                                           --reverse_checkpoint <your_path>             \
                                                           --forward_checkpoint <your_path>             \
                                                           --teacher_checkpoint <your_path>             \
                                                           --num_reverse_cons_steps 4                   \
                                                           --reverse_timesteps 259 519 779 999          \
                                                           --num_forward_cons_steps 4                   \
                                                           --forward_timesteps 19 259 519 779           \
                                                           --path_to_prompts benchmarks/instructions/editing_pie_bench_140.csv         \
                                                           --path_to_images benchmarks/images/pie_bench_140_images             \
                                                           --guidance_scale 19                          \
                                                           --dynamic_guidance True                      \
                                                           --tau1 0.8                                   \
                                                           --tau2 0.8                                   \
                                                           --cross_replace_steps $1                     \
                                                           --self_replace_steps $2                      \
                                                           --amplify_factor 4                           \
                                                           --max_forward_timestep_idx 49                \
                                                           --start_timestep 19                          \
                                                           --nti_guidance_scale 8.0                     \
                                                           --use_npi False                              \
                                                           --use_nti False                              \
                                                           --use_cons_inversion True                    \
                                                           --use_cons_editing True                      \
                                                           --lora_rank 64                               \
                                                           --w_embed_dim 512                            \
                                                           --num_ddim_steps 50                          \
                                                           --device cuda                                \
                                                           --saving_dir results_editing_cons/results_${1},${2} \
                                                           --dtype fp32                                 \
                                                           --is_replacement False                       \
                                                           --seed 30
done
