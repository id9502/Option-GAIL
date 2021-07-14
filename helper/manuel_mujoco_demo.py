#!/usr/bin/env python3
import os
import sys
import torch
from model.option_policy import Policy, OptionPolicy
from utils.state_filter import StateFilter
from envir.mujoco_env import MujocoEnv
from default_config import mujoco_config


def option_loop(env, policy, state_filter, fixed):
    with torch.no_grad():
        a_array = []
        s_array = []
        r_array = []
        s, done = env.reset(random=not fixed), False
        ct = torch.empty(1, 1, dtype=torch.long, device=policy.device).fill_(policy.dim_c)
        while not done:
            st = torch.as_tensor(s.copy(), dtype=torch.float32, device=policy.device).unsqueeze(0)
            s = torch.as_tensor(state_filter(s, fixed), dtype=torch.float32, device=policy.device).unsqueeze(0)
            ct = policy.sample_option(s, ct, fixed=fixed).detach()
            at = policy.sample_action(s, ct, fixed=fixed).detach()
            s_array.append(st)
            a_array.append(at)
            s, r, done = env.step(at.cpu().squeeze(dim=0).numpy())
            r_array.append(r)
        a_array = torch.cat(a_array, dim=0)
        s_array = torch.cat(s_array, dim=0)
        r_array = torch.as_tensor(r_array, dtype=torch.float32, device=policy.device).unsqueeze(dim=-1)
    return s_array, a_array, r_array


def loop(env, policy, state_filter, fixed):
    with torch.no_grad():
        a_array = []
        s_array = []
        r_array = []
        s, done = env.reset(random=not fixed), False
        while not done:
            st = torch.as_tensor(s.copy(), dtype=torch.float32, device=policy.device).unsqueeze(0)
            s = torch.as_tensor(state_filter(s, fixed), dtype=torch.float32, device=policy.device).unsqueeze(0)
            at = policy.sample_action(s, fixed=fixed).detach()
            s_array.append(st)
            a_array.append(at)
            s, r, done = env.step(at.cpu().squeeze(dim=0).numpy())
            r_array.append(r)
        a_array = torch.cat(a_array, dim=0)
        s_array = torch.cat(s_array, dim=0)
        r_array = torch.as_tensor(r_array, dtype=torch.float32, device=policy.device).unsqueeze(dim=-1)
    return s_array, a_array, r_array


def get_demo(env_name, n_demo=409600, display=False):
    expert_path = f"./data/ppo-expert/{env_name}_expert.torch"
    config_path = f"./data/ppo-expert/{env_name}_expert.log"
    if os.path.isfile(config_path):
        mujoco_config.load_saved(config_path)

    mujoco_config.device = "cpu"
    use_rs = mujoco_config.use_state_filter
    use_option = mujoco_config.use_option

    env = MujocoEnv(env_name)
    dim_s, dim_a = env.state_action_size()
    env.init(display=display)

    policy_state, filter_state = torch.load(expert_path, map_location="cpu")
    policy = OptionPolicy(mujoco_config, dim_s, dim_a) if use_option else Policy(mujoco_config, dim_s, dim_a)
    policy.load_state_dict(policy_state)
    rs = StateFilter(enable=use_rs)
    rs.load_state_dict(filter_state)

    sample = []
    n_current_demo = 0
    while n_current_demo < n_demo:
        s_array, a_array, r_array = option_loop(env, policy, rs, fixed=True) if use_option else loop(env, policy, rs, fixed=True)
        print(f"R-Sum={r_array.sum()}, L={r_array.size(0)}")
        keep = input(f"{n_current_demo}/{n_demo} Keep this ? [y|n]>>>")
        if keep == 'y':
            sample.append((s_array, a_array, r_array))
            n_current_demo += r_array.size(0)
            print(f"Current: {n_current_demo} / {n_demo}")
    torch.save((sample, rs.state_dict()), f"./data/{env_name}_sample.torch")
    print(f"saved @ ./data/{env_name}_sample.torch")


if __name__ == "__main__":
    env_name = "AntPush-v0"
    n_demo = 409600
    if len(sys.argv) > 1:
        env_name = sys.argv[1]
    if len(sys.argv) > 2:
        n_demo = int(sys.argv[2])
    get_demo(env_name, n_demo)
