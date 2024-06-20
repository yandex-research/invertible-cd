## Table of contents

* [Generation instruction](#instructions-for-generation)
* [Generation SDXL](#generation-with-icd-distilled-from-sdxl)
* [Generation SD1.5](#generation-with-icd-distilled-from-sd15)
* [Editing instruction](#instructions-for-editing)
* [Editing SD1.5](#editing-with-icd-distilled-from-sd15-using-prompt-2-prompt-approach)
* [Editing SDXL](#editing-with-icd-distilled-from-sdxl)

## Instructions for generation
**Preparation**:
1. Specify the models paths (*reverse, forward, teacher checkpoints*) in the ``sdxl/launch_generation_iCD_sdxl.sh`` and ``sd1.5/launch_generation_iCD_sd1.5.sh``.
2. Specify the number of steps for iCD (*num_reverse_cons_steps*, *num_forward_cons_steps*) and time steps (*reverse_timesteps*, *forward_timesteps*) according to the chosen model.
3. Specify path to the validation prompts. Below we use ``benchmarks/instructions/parti-prompts-eval.csv`` 
and ``benchmarks/instructions/coco.csv`` 

**Hyperparameters**:

The main hyperparameters are *tau1* and *tau2*, which are responsible for dynamic guidance. 
Their values can be varied from 0.0 to 1.0, with 1.0 meaning constant guidance and 0.0 meaning no guidance generation.
In our work, we use *tau1=tau2=tau*, which corresponds to a step function (if *t<tau* iCD follows unguided sampling and sets the initial guidance scale for *t>=tau*).
We find that *tau=0.8* provide diverse and high-quality samples.  

Please, see Figure 2a in the paper for more details.
You can also play with *guidance_scale*.

## Generation with iCD-XL
```
export PATH="<path_to_main_root>:$PATH" # Specify path if needed
export PYTHONPATH=<path_to_main_root> # Specify path if needed
. sdxl/launch_generation_iCD_sdxl.sh
```
We expect the following results:
<p align="center">
<img src="../assets/uncurated_samples_sdxl.jpg" width="1080px"/>
</p>

## Generation with iCD-SD1.5

```
. sd1.5/launch_generation_sd1.5.sh
```
We expect the following results:
<p align="center">
<img src="../assets/uncurated_samples.jpg" width="1080px"/>
</p>

## Instructions for editing
**Preparation**:
1. Specify the models paths (*reverse, forward, teacher checkpoints*) in the ``sdxl/launch_editing_iCD_sdxl.sh`` and ``sd1.5/launch_editing_iCD_sd1.5.sh``.
2. Specify the number of steps for iCD (*num_reverse_cons_steps*, *num_forward_cons_steps*) and time steps (*reverse_timesteps*, *forward_timesteps*) according to the chosen model.
3. Specify path to the editing instructions. Below we use ``benchmarks/instructions/editing_pie_bench_140.csv``.
4. [Download](https://storage.yandexcloud.net/yandex-research/invertible-cd/editing_images.tar) the images for editing and put them to ``benchmarks/images``. Below we use ``benchmarks/images/pie_bench_140_images``

**Hyperparameters**:

For the SD1.5, we use [prompt-2-prompt](https://arxiv.org/abs/2208.01626) using two main hyperparameters:
*cross_replace_steps* and *self_replace_steps*. They provide the control for editing-preservation trade-off.
Below we use *cross_replace_steps=0.3*, *self_replace_steps=0.6*. Unfortunately, results are highly sensitive to
the hyperparameters, so you need to try different values to find the best one.

In the editing, the use of dynamic guidance is crucial for preserving the reference image.
Thus, we use dynamic guidance and set *tau1=tau2=0.8* or *tau1=tau2=0.7* depending on the configuration.
You should try both to find the best one.

You can also play with the *amplify_factor* and *guidance_scale*.

## Editing with iCD-SD1.5 using prompt-2-prompt approach
In our experiments, we mainly consider the model using reverse: [259, 519, 779, 999]; forward: [19, 259, 519, 779]
timesteps and τ=0.8
```
. sd1.5/launch_editing_iCD_sd1.5.sh
```
We also present the running scripts for the baselines ([NPI](https://arxiv.org/abs/2305.16807v1) and [NTI](https://null-text-inversion.github.io/))
```
. sd1.5/launch_editing_NPI_sd1.5.sh

. sd1.5/launch_editing_NTI_sd1.5.sh
```

## Editing with iCD-XL
In our experiments, we mainly consider the model using reverse: [249, 499, 699, 999]; forward: [19, 249, 499, 699]
timesteps and τ=0.7.
```
. bash sdxl/launch_editing_iCD_sdxl.sh
```
