#!/usr/bin/env python3

import os
import torch
from model.option_ppo import MoEPPO
from model.option_gail import MoEGAIL
from utils.utils import lr_factor_func, reward_validate, get_dirs, set_seed
from utils.agent import Sampler
from utils.logger import Logger
from utils.config import ARGConfig, Config
from default_config import mujoco_config, rlbench_config


def make_gail(config: Config, dim_s, dim_a):
    gail = MoEGAIL(config, dim_s=dim_s, dim_a=dim_a)
    ppo = MoEPPO(config, gail.policy)
    return gail, ppo


def train_g(ppo: MoEPPO, sample_sxar, factor_lr):
    ppo.step(sample_sxar, lr_mult=factor_lr)


def train_d(gail: MoEGAIL, sample_sxar, demo_sxar, n_step=10):
    return gail.step(sample_sxar, demo_sxar, n_step=n_step)


def sample_batch(gail: MoEGAIL, agent, n_sample, demo_sa_array):
    demo_sa_in = agent.filter_demo(demo_sa_array)
    sample_sar_in = agent.collect(gail.policy.state_dict(), n_sample, fixed=False)
    sample_sar, sample_rsum = gail.convert_sample(sample_sar_in)
    demo_sar, demo_rsum = gail.convert_demo(demo_sa_in)
    return sample_sar, demo_sar, sample_rsum, demo_rsum


def learn(config: Config, msg="default"):
    env_type = config.env_type
    if env_type == "mujoco":
        from envir.mujoco_env import MujocoEnv as Env, get_demo
    elif env_type == "rlbench":
        from envir.rlbench_env import RLBenchEnv as Env, get_demo
    else:
        raise ValueError(f"Unknown env type {env_type}")

    n_demo = config.n_demo
    n_sample = config.n_sample
    n_thread = config.n_thread
    n_epoch = config.n_epoch
    seed = config.seed
    env_name = config.env_name
    use_state_filter = config.use_state_filter

    set_seed(seed)

    log_dir, save_dir, sample_name, pretrain_name = get_dirs(seed, "gail-moe", env_type, env_name, msg, is_opt=False)
    with open(os.path.join(save_dir, "config.log"), 'w') as f:
        f.write(str(config))
    logger = Logger(log_dir)
    save_name_f = lambda i: os.path.join(save_dir, f"gail_{i}.torch")

    env = Env(env_name)
    dim_s, dim_a = env.state_action_size()
    demo, _ = get_demo(path=sample_name, n_demo=n_demo, display=False)

    gail, ppo = make_gail(config, dim_s=dim_s, dim_a=dim_a)
    sampling_agent = Sampler(seed, env, gail.policy, use_state_filter=use_state_filter, n_thread=n_thread)

    demo_sa_array = tuple((s.to(gail.device), a.to(gail.device)) for s, a, r in demo)

    sample_sxar, demo_sxar, sample_r, demo_r = sample_batch(gail, sampling_agent, n_sample, demo_sa_array)
    info_dict, cs_sample = reward_validate(sampling_agent, gail.policy, do_print=True)

    logger.log_test_info(info_dict, 0)
    print(f"init: r-sample-avg={sample_r}, d-demo-avg={demo_r} ; {msg}")

    for i in range(n_epoch):
        sample_sar, demo_sxar, sample_r, demo_r = sample_batch(gail, sampling_agent, n_sample, demo_sa_array)

        train_d(gail, sample_sar, demo_sxar)
        # factor_lr = lr_factor_func(i, 1000., 1., 0.0001)
        train_g(ppo, sample_sar, factor_lr=1.)
        if (i + 1) % 20 == 0:
            info_dict, cs_sample = reward_validate(sampling_agent, gail.policy, do_print=True)

            torch.save((gail.state_dict(), sampling_agent.state_dict()), save_name_f(i))
            logger.log_test_info(info_dict, i)
            print(f"{i}: r-sample-avg={sample_r}, d-demo-avg={demo_r} ; {msg}")
        else:
            print(f"{i}: r-sample-avg={sample_r}, d-demo-avg={demo_r} ; {msg}")
        logger.log_train("r-sample-avg", sample_r, i)
        logger.log_train("r-demo-avg", demo_r, i)
        logger.flush()


if __name__ == "__main__":
    import torch.multiprocessing as multiprocessing
    multiprocessing.set_start_method('spawn')

    arg = ARGConfig()
    arg.add_arg("env_type", "mujoco", "Environment type, can be [mujoco, rlbench, mini]")
    arg.add_arg("env_name", "Walker2d-v2", "Environment name")
    arg.add_arg("device", "cuda:0", "Computing device")
    arg.add_arg("tag", "default", "Experiment tag")
    arg.add_arg("n_demo", 1000, "Number of demonstration s-a")
    arg.add_arg("n_epoch", 4000, "Number of training epochs")
    arg.add_arg("dim_c", 4, "Number of options")
    arg.add_arg("seed", torch.randint(100, ()).item(), "Random seed")
    arg.add_arg("use_state_filter", True, "Use state filter")
    arg.parser()

    if arg.env_type == "rlbench":
        config = rlbench_config
    elif arg.env_type == "mujoco":
        config = mujoco_config
    else:
        raise ValueError("mini for circle env; rlbench for rlbench env; mujoco for mujoco env")

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

    print(f">>>> Training MoE-GAIL using {config.env_name} environment on {config.device}")

    learn(config, msg=config.tag)
