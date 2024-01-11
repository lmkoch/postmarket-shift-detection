#!/bin/bash

cd /mnt/qb/work/berens/lkoch54/devel/subgroup-shift-detection

############################
# Task classifiers

# python scripts/train_task_classifier.py --config_file ./config/lightning/eyepacs.yaml --method muks --data_frac 1.0  --exp_dir ./experiments/nov/task/${dataset} --slurm



##############################################################
# All experiments with fixed (limited) train sample size:
# 100, 500, 1000, 5000, 10000

exp_base_dir="./experiments/eyepacs_seeded_trainset_limited_debug"

# exp_dir=$exp_base_dir/eyepacs_quality
# size=500
# seed=1000

# test="c2st"
# config_file="./config/lightning/eyepacs_${test}_general.yaml"
# method=${test}

# # python scripts/train_lightning_task.py --config_file ${config_file} --seed ${seed} --size ${size} --method ${method} --exp_dir ${exp_dir} --slurm
# python scripts/train_lightning_task.py --config_file ${config_file} --subset_qual Adequate  --seed ${seed} --train_data_abs_size ${size} --method ${method} --exp_dir ${exp_dir} --slurm


# quality - ID / OOD
# all other subgroups

# C2ST / MMDD / MUKS

# no ablations yet

# # Re-run because this had failed:
# test="muks"
# size=500
# seed=1001
# ii="Asian"
# exp_dir=$exp_base_dir/eyepacs_ethnicity

# config_file="./config/general/eyepacs_${test}_general.yaml"
# method=${test}

# python scripts/train_lightning_task.py --config_file ${config_file} --subset_ethnicity ${ii}  --seed ${seed} --train_data_abs_size ${size} --method ${method} --exp_dir ${exp_dir} --slurm  


# # for size in 100 500 1000 2000 5000 10000;
# for size in 100 2000 5000 10000;
# # for size in 500 1000;
# do
#     # for seed in 1000 1001 1002 1003 1004;
#     for seed in 1000 1001;
#     do

#         for test in muks c2st mmdd;
#         do
#             config_file="./config/general/eyepacs_${test}_general.yaml"
#             method=${test}

#             # ##########################
#             # # Eyepacs co-morbidities

#             exp_dir=$exp_base_dir/eyepacs_comorb
#             python scripts/train_lightning_task.py --config_file ${config_file} --subset_comorbid True  --seed ${seed} --train_data_abs_size ${size} --method ${method} --exp_dir ${exp_dir} --slurm
        
#             ##########################
#             # Eyepacs quality

#             exp_dir=$exp_base_dir/eyepacs_quality
#             python scripts/train_lightning_task.py --config_file ${config_file} --subset_qual Adequate  --seed ${seed} --train_data_abs_size ${size} --method ${method} --exp_dir ${exp_dir} --slurm

#             ###########################
#             # Eyepacs sex

#             exp_dir=$exp_base_dir/eyepacs_sex
#             python scripts/train_lightning_task.py --config_file ${config_file} --subset_sex Female  --seed ${seed} --train_data_abs_size ${size} --method ${method} --exp_dir ${exp_dir} --slurm

#             ###########################
#             # Eyepacs ethnicity
#             exp_dir=$exp_base_dir/eyepacs_ethnicity

#             for ii in AfricanDescent Asian Caucasian Indiansubcontinentorigin LatinAmerican Multi-racial NativeAmerican;
#             do
#                 python scripts/train_lightning_task.py --config_file ${config_file} --subset_ethnicity ${ii}  --seed ${seed} --train_data_abs_size ${size} --method ${method} --exp_dir ${exp_dir} --slurm           
#             done

#             ###########################
#             # Eyepacs quality OOD
#             exp_dir=$exp_base_dir/eyepacs_quality_OOD
#             python scripts/train_lightning_task.py --config_file ${config_file} --subset_qual Insufficient  --seed ${seed} --train_data_abs_size ${size} --method ${method} --exp_dir ${exp_dir} --slurm


#         done

#     done
# # size
# done

# ## For sex differences, need larger sample size
# # for size in 100 500 1000 2000 5000 10000;
# # for size in 100 2000 5000 10000;
# for size in 20000 30000 40000;
# do
#     # for seed in 1000 1001 1002 1003 1004;
#     for seed in 1000 1001;
#     do

#         for test in muks c2st mmdd;
#         do
#             config_file="./config/general/eyepacs_${test}_general.yaml"
#             method=${test}

#             ###########################
#             # Eyepacs sex

#             exp_dir=$exp_base_dir/eyepacs_sex
#             python scripts/train_lightning_task.py --config_file ${config_file} --subset_sex Female  --seed ${seed} --train_data_abs_size ${size} --method ${method} --exp_dir ${exp_dir} --slurm

#         done
#     done
# done

##############################################################
# 2023-07-11
# C2ST-96


size=1000
test="c2st"

# # for seed in 1000 1001 1002 1003 1004;
# for seed in 1000 1001;
# do
#     config_file="./config/general/eyepacs_${test}_general.yaml"
#     method=${test}

#     # ##########################
#     # # Eyepacs co-morbidities

#     exp_dir=$exp_base_dir/c2st96/eyepacs_comorb
#     python scripts/train_lightning_task.py --img_size 96 --batch_size 16  --config_file ${config_file} --subset_comorbid True  --seed ${seed} --train_data_abs_size ${size} --method ${method} --exp_dir ${exp_dir} --slurm

#     ##########################
#     # Eyepacs quality

#     exp_dir=$exp_base_dir/c2st96/eyepacs_quality
#     python scripts/train_lightning_task.py --img_size 96 --batch_size 16  --config_file ${config_file} --subset_qual Adequate  --seed ${seed} --train_data_abs_size ${size} --method ${method} --exp_dir ${exp_dir} --slurm

#     ###########################
#     # Eyepacs sex

#     exp_dir=$exp_base_dir/c2st96/eyepacs_sex
#     python scripts/train_lightning_task.py --img_size 96 --batch_size 16  --config_file ${config_file} --subset_sex Female  --seed ${seed} --train_data_abs_size ${size} --method ${method} --exp_dir ${exp_dir} --slurm

#     ###########################
#     # Eyepacs ethnicity
#     exp_dir=$exp_base_dir/c2st96/eyepacs_ethnicity

#     for ii in AfricanDescent Asian Caucasian Indiansubcontinentorigin LatinAmerican Multi-racial NativeAmerican;
#     do
#         python scripts/train_lightning_task.py --img_size 96 --batch_size 16  --config_file ${config_file} --subset_ethnicity ${ii}  --seed ${seed} --train_data_abs_size ${size} --method ${method} --exp_dir ${exp_dir} --slurm           
#     done

#     ###########################
#     # Eyepacs quality OOD
#     exp_dir=$exp_base_dir/c2st96/eyepacs_quality_OOD
#     python scripts/train_lightning_task.py --img_size 96 --batch_size 16  j--config_file ${config_file} --subset_qual Insufficient  --seed ${seed} --train_data_abs_size ${size} --method ${method} --exp_dir ${exp_dir} --slurm

# done



# # C2ST arch ablations

# ############################
# # C2ST with shallow classifier: eyepacs quality

# for seed in 1000 1001;
# do
#     config_file="./config/general/eyepacs_${test}_general.yaml"
#     method=${test}

#     ##########################
#     # Eyepacs quality

#     exp_dir=$exp_base_dir/c2st_arch/eyepacs_quality
#     python scripts/train_lightning_task.py --c2st_arch shallow --config_file ${config_file} --subset_qual Adequate  --seed ${seed} --train_data_abs_size ${size} --method ${method} --exp_dir ${exp_dir} --slurm
#     python scripts/train_lightning_task.py --c2st_arch resnet18 --config_file ${config_file} --subset_qual Adequate  --seed ${seed} --train_data_abs_size ${size} --method ${method} --exp_dir ${exp_dir} --slurm
# done


# Gradual over-representation

# ##############################
# # quality gradual overrepresentation


# for seed in 1000 1001;
# do

#     for test in muks c2st mmdd;
#     do
#         config_file="./config/general/eyepacs_${test}_general.yaml"
#         method=${test}


#         for factor in 1 5 10 100;
#         do


#             ##########################
#             # Eyepacs quality

#             exp_dir=$exp_base_dir/gradual/eyepacs_quality
#             python scripts/train_lightning_task.py --subset_qual_gradual ${factor}  --config_file ${config_file} --seed ${seed} --train_data_abs_size ${size} --method ${method} --exp_dir ${exp_dir} --slurm

#         done
#     done
# done





# ##########################
# # MMDD ablations (old)

# # data augmentation: yes / no
# # backbone architecture: liu / resnet
# python scripts/train_lightning_task.py --data_aug --mmd_feature_extractor liu  --config_file ./config/lightning/eyepacs_quality_mmdd.yaml --method mmdd --data_frac 1.0  --exp_dir ./experiments/eyepacs/eyepacs_quality_mmd_ablation --slurm
# python scripts/train_lightning_task.py --data_aug --mmd_feature_extractor resnet50  --config_file ./config/lightning/eyepacs_quality_mmdd.yaml --method mmdd --data_frac 1.0  --exp_dir ./experiments/eyepacs/eyepacs_quality_mmd_ablation --slurm
# python scripts/train_lightning_task.py --no_data_aug --mmd_feature_extractor liu  --config_file ./config/lightning/eyepacs_quality_mmdd.yaml --method mmdd --data_frac 1.0  --exp_dir ./experiments/eyepacs/eyepacs_quality_mmd_ablation --slurm
# python scripts/train_lightning_task.py --no_data_aug --mmd_feature_extractor resnet50  --config_file ./config/lightning/eyepacs_quality_mmdd.yaml --method mmdd --data_frac 1.0  --exp_dir ./experiments/eyepacs/eyepacs_quality_mmd_ablation --slurm



################################
# Revision: revisit classifier results

# python scripts/train_task_classifier.py --config_file ./config/lightning/eyepacs.yaml --method muks --data_frac 1.0  --exp_dir ./experiments/nov/task/${dataset} --slurm



##############################################################
# All experiments with fixed (limited) train sample size:
# 100, 500, 1000, 5000, 10000

exp_dir="./experiments/revisions"
method="muks"
config_file="./config/general/eyepacs_${method}_general.yaml"
seed=1000

python scripts/train_lightning_task.py --config_file ${config_file}  --seed ${seed}  --method ${method} --exp_dir ${exp_dir} --slurm  
