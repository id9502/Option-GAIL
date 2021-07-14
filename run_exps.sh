#!/usr/bin/env bash

###### prerequisites: ######
# python3 >= 3.6
#     torch >= 1.7
#     tensorboard >= 2.2
#     matplotlib >= 3.2

###### for Hopper-v2 #####
# Option-GAIL:
python3 ./option_gail_learn.py --env_type mujoco --env_name Hopper-v2 --use_option True --use_c_in_discriminator True --n_demo 1000 --device "cuda:0" --use_state_filter True --tag option-gail-1k

# GAIL-HRL:
python3 ./option_gail_learn.py --env_type mujoco --env_name Hopper-v2 --use_option True --use_c_in_discriminator False --n_demo 1000 --device "cuda:0" --use_state_filter True --tag gail-hrl-1k

# GAIL:
python3 ./option_gail_learn.py --env_type mujoco --env_name Hopper-v2 --use_option False --use_c_in_discriminator False --n_demo 1000 --device "cuda:0" --use_state_filter True --tag gail-1k

# H-BC:
python3 ./option_bc_learn.py --env_type mujoco --env_name Hopper-v2 --use_option True --loss_type MLE --n_demo 1000 --device "cuda:0" --use_state_filter False --tag hbc-1k

# BC:
python3 ./option_bc_learn.py --env_type mujoco --env_name Hopper-v2 --use_option False --loss_type L2 --n_demo 1000 --device "cuda:0" --use_state_filter False --tag bc-1k

# Pre-train:
python3 ./option_gail_learn.py --env_type mujoco --env_name Hopper-v2 --use_option True --use_c_in_discriminator False --use_d_info_gail True --train_option False --n_pretrain_epoch 50 --n_demo 1000 --device "cuda:0" --use_state_filter True --tag d_info_gail-1k

# MoE:
python3 ./option_gail_learn_moe.py --env_type mujoco --env_name Hopper-v2 --n_demo 1000 --device "cuda:0" --use_state_filter True --tag gail-moe-1k


###### for Walker2d-v2 #####
# Option-GAIL:
python3 ./option_gail_learn.py --env_type mujoco --env_name Walker2d-v2 --use_option True --use_c_in_discriminator True --n_demo 5000 --device "cuda:0" --use_state_filter True --tag option-gail-5k

# GAIL-HRL:
python3 ./option_gail_learn.py --env_type mujoco --env_name Walker2d-v2 --use_option True --use_c_in_discriminator False --n_demo 5000 --device "cuda:0" --use_state_filter True --tag gail-hrl-5k

# GAIL:
python3 ./option_gail_learn.py --env_type mujoco --env_name Walker2d-v2 --use_option False --use_c_in_discriminator False --n_demo 5000 --device "cuda:0" --use_state_filter True --tag gail-5k

# H-BC:
python3 ./option_bc_learn.py --env_type mujoco --env_name Walker2d-v2 --use_option True --loss_type MLE --n_demo 5000 --device "cuda:0" --use_state_filter False --tag hbc-5k

# BC:
python3 ./option_bc_learn.py --env_type mujoco --env_name Walker2d-v2 --use_option False --loss_type L2 --n_demo 5000 --device "cuda:0" --use_state_filter False --tag bc-5k

# Pre-train:
python3 ./option_gail_learn.py --env_type mujoco --env_name Walker2d-v2 --use_option True --use_c_in_discriminator False --use_d_info_gail True --train_option False --n_pretrain_epoch 50 --n_demo 5000 --device "cuda:0" --use_state_filter True --tag d_info_gail-5k

# MoE:
python3 ./option_gail_learn_moe.py --env_type mujoco --env_name Walker2d-v2 --n_demo 5000 --device "cuda:0" --use_state_filter True --tag gail-moe-5k


###### for AntPush-v0 #####
# Option-GAIL:
python3 ./option_gail_learn.py --env_type mujoco --env_name AntPush-v0 --use_option True --use_c_in_discriminator True --n_demo 50000 --device "cuda:0" --use_state_filter False --tag option-gail-50k

# GAIL-HRL:
python3 ./option_gail_learn.py --env_type mujoco --env_name AntPush-v0 --use_option True --use_c_in_discriminator False --n_demo 50000 --device "cuda:0" --use_state_filter False --tag gail-hrl-50k

# GAIL:
python3 ./option_gail_learn.py --env_type mujoco --env_name AntPush-v0 --use_option False --use_c_in_discriminator False --n_demo 50000 --device "cuda:0" --use_state_filter False --tag gail-50k

# H-BC:
python3 ./option_bc_learn.py --env_type mujoco --env_name AntPush-v0 --use_option True --loss_type MLE --n_demo 50000 --device "cuda:0" --use_state_filter False --tag hbc-50k

# BC:
python3 ./option_bc_learn.py --env_type mujoco --env_name AntPush-v0 --use_option False --loss_type L2 --n_demo 50000 --device "cuda:0" --use_state_filter False --tag bc-50k

# Pre-train:
python3 ./option_gail_learn.py --env_type mujoco --env_name AntPush-v0 --use_option True --use_c_in_discriminator False --use_d_info_gail True --train_option False --n_pretrain_epoch 100 --n_demo 50000 --device "cuda:0" --use_state_filter True --tag d_info_gail-5k

# MoE:
python3 ./option_gail_learn_moe.py --env_type mujoco --env_name AntPush-v0 --n_demo 50000 --device "cuda:0" --use_state_filter False --tag gail-moe-50k


###### for CloseMicrowave2 #####
# Option-GAIL:
python3 ./option_gail_learn.py --env_type rlbench --env_name CloseMicrowave2 --use_option True --use_c_in_discriminator True --n_demo 1000 --device "cuda:0" --use_state_filter True --tag option-gail-1k

# GAIL-HRL:
python3 ./option_gail_learn.py --env_type rlbench --env_name CloseMicrowave2 --use_option True --use_c_in_discriminator False --n_demo 1000 --device "cuda:0" --use_state_filter True --tag gail-hrl-1k

# GAIL:
python3 ./option_gail_learn.py --env_type rlbench --env_name CloseMicrowave2 --use_option False --use_c_in_discriminator False --n_demo 1000 --device "cuda:0" --use_state_filter True --tag gail-1k

# H-BC:
python3 ./option_bc_learn.py --env_type rlbench --env_name CloseMicrowave2 --use_option True --loss_type MLE --n_demo 1000 --device "cuda:0" --use_state_filter False --tag hbc-1k

# BC:
python3 ./option_bc_learn.py --env_type rlbench --env_name CloseMicrowave2 --use_option False --loss_type L2 --n_demo 1000 --device "cuda:0" --use_state_filter False --tag bc-1k

# Pre-train:
python3 ./option_gail_learn.py --env_type rlbench --env_name CloseMicrowave2 --use_option True --use_c_in_discriminator False --use_d_info_gail True --train_option False --n_pretrain_epoch 50 --n_demo 1000 --device "cuda:0" --use_state_filter True --tag d_info_gail-1k

# MoE:
python3 ./option_gail_learn_moe.py --env_type rlbench --env_name CloseMicrowave2 --n_demo 1000 --device "cuda:0" --use_state_filter True --tag gail-moe-1k


