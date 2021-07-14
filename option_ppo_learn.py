#!/usr/bin/env python3

import os
import torch
from model.option_ppo import PPO, OptionPPO
from model.option_policy import OptionPolicy, Policy
from utils.agent import Sampler
from utils.utils import lr_factor_func, sample_batch, get_dirs, reward_validate, set_seed
from utils.logger import Logger
import matplotlib.pyplot as plt
from utils.config import Config, ARGConfig
from default_config import mujoco_config, rlbench_config


def learn(config: Config, msg="default"):
    env_type = config.env_type
    if env_type == "mujoco":
        from envir.mujoco_env import MujocoEnv as Env
    else:
        from envir.rlbench_env import RLBenchEnv as Env

    use_option = config.use_option
    env_name = config.env_name
    n_sample = config.n_sample
    n_thread = config.n_thread
    n_epoch = config.n_epoch
    seed = config.seed
    use_state_filter = config.use_state_filter

    set_seed(seed)

    log_dir, save_dir, sample_name, pretrain_name = get_dirs(seed, "ppo", "mujoco", env_name, msg, use_option)
    with open(os.path.join(save_dir, "config.log"), 'w') as f:
        f.write(str(config))
    logger = Logger(log_dir)

    save_name_f = lambda i: os.path.join(save_dir, f"{i}.torch")

    env = Env(env_name)
    dim_s, dim_a = env.state_action_size()

    if use_option:
        policy = OptionPolicy(config, dim_s=dim_s, dim_a=dim_a)
        ppo = OptionPPO(config, policy)
    else:
        policy = Policy(config, dim_s=dim_s, dim_a=dim_a)
        ppo = PPO(config, policy)

    sampling_agent = Sampler(seed, env, policy, use_state_filter=use_state_filter, n_thread=n_thread)

    for i in range(n_epoch):
        sample_sxar, sample_r = sample_batch(policy, sampling_agent, n_sample)
        lr_mult = lr_factor_func(i, n_epoch, 1., 0.)
        ppo.step(sample_sxar, lr_mult=lr_mult)
        if (i + 1) % 50 == 0:
            info_dict, cs_sample = reward_validate(sampling_agent, policy)

            if cs_sample is not None:
                a = plt.figure()
                a.gca().plot(cs_sample[0][1:])
                logger.log_test_fig("sample_c", a, i)
            torch.save((policy.state_dict(), sampling_agent.state_dict()), save_name_f(i))
            logger.log_test_info(info_dict, i)
        print(f"{i}: r-sample-avg={sample_r} ; {msg}")
        logger.log_train("r-sample-avg", sample_r, i)
        logger.flush()


if __name__ == "__main__":
    import torch.multiprocessing as multiprocessing
    multiprocessing.set_start_method('spawn')

    arg = ARGConfig()
    arg.add_arg("use_option", True, "Use Option when training or not")
    arg.add_arg("env_type", "mujoco", "Environment type, can be [mujoco, rlbench, mini]")
    arg.add_arg("env_name", "Walker2d-v2", "Environment name")
    arg.add_arg("device", "cuda:0", "Computing device")
    arg.add_arg("tag", "default", "Experiment tag")
    arg.add_arg("n_epoch", 3000, "Number of training epochs")
    arg.add_arg("seed", torch.randint(100, ()).item(), "Random seed")
    arg.parser()

    config = mujoco_config if arg.env_type == "mujoco" else rlbench_config
    config.update(arg)

    if config.env_name.startswith("Humanoid"):
        config.hidden_policy = (512, 512)
        config.hidden_critic = (512, 512)
        print(f"Training Humanoid.* envs with larger policy network size :{config.hidden_policy}")
    if config.env_type == "rlbench":
        config.hidden_policy = (128, 128)
        config.hidden_option = (128, 128)
        config.hidden_critic = (128, 128)
        config.log_clamp_policy = (-20., -2.)
        print(f"Training RLBench.* envs with larger policy network size :{config.hidden_policy}")

    print(f">>>> Training {'Option-' if config.use_option else ''}PPO on {config.env_name} environment, on {config.device}")
    learn(config, msg=config.tag)
