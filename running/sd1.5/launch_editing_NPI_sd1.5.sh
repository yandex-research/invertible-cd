####################################################################
# Editing with NPI for different hyperparameters
####################################################################

for crs_srs in "0.2 0.5"
do
    set -- $crs_srs
    echo "NPI: cross_replace_steps: $1, self_replace_steps $2"
    CUDA_VISIBLE_DEVICES=0 python3.9 sd1.5/edit.py --model_id_DM runwayml/stable-diffusion-v1-5  \
                                                           --path_to_prompts  benchmarks/instructions/editing_pie_bench_140.csv        \
                                                           --path_to_images benchmarks/images/pie_bench_140_images             \
                                                           --lora_rank 64                               \
                                                           --w_embed_dim 0                              \
                                                           --num_ddim_steps 50                          \
                                                           --max_forward_timestep_idx 49                \
                                                           --start_timestep 19                          \
                                                           --use_cons_inversion False                   \
                                                           --nti_guidance_scale 8.0                     \
                                                           --use_npi True                               \
                                                           --use_nti False                              \
                                                           --use_cons_editing False                     \
                                                           --cross_replace_steps $1                     \
                                                           --self_replace_steps $2                      \
                                                           --amplify_factor 3                           \
                                                           --guidance_scale 8.0                         \
                                                           --device cuda                                \
                                                           --saving_dir results_npi_editing             \
                                                           --dtype fp32                                 \
                                                           --is_replacement False                       \
                                                           --seed 30
done
