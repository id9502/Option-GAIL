from envir.rlbench_env import RLBenchEnv, tasks
from utils.state_filter import StateFilter
from rlbench.task_environment import _DT, Quaternion
from rlbench.action_modes import ArmActionMode, ActionMode
import numpy as np
import random as rnd
import torch
import sys


def manuel_demo(env_name, n_demo=1):
    env = RLBenchEnv(env_name).init(display=True)
    demos = []
    while len(demos) < n_demo:
        demo = env.gen_demo(random=False)
        if input(f"R={demo[-1].sum().item()} Keep this ? >>>") == "y":
            demos.append(demo)
    return demos


def manuel_demo2(env_name, n_demo=10, n_repeat=2):
    random = True
    env = RLBenchEnv(env_name).init(display=True)
    env.task._static_positions = not random

    demos = []
    while len(demos) < n_demo * n_repeat:
        seed = rnd.randint(0, 4096) if random else 0
        np.random.seed(seed)
        pack = env.task.get_demos(1, True)[0]
        if input("Keep this as expert ground-truth path ? [y/n]>>>") != 'y':
            continue
        demo_sub = []
        while len(demo_sub) < n_repeat:
            s_array = []
            a_array = []
            r_array = []
            np.random.seed(seed)
            desc, obs = env.task.reset()
            for o_tar in pack[1:]:
                action = []
                if env._action_config.arm == ArmActionMode.ABS_JOINT_VELOCITY:
                    action.extend((o_tar.joint_positions - obs.joint_positions) / _DT)
                elif env._action_config.arm == ArmActionMode.ABS_JOINT_POSITION:
                    action.extend(o_tar.joint_positions)
                elif env._action_config.arm == ArmActionMode.ABS_JOINT_TORQUE:
                    action.extend(o_tar.joint_forces)
                    raise TypeError("Warning, abs_joint_torque is not currently supported")
                elif env._action_config.arm == ArmActionMode.ABS_EE_POSE_WORLD_FRAME:
                    action.extend(o_tar.gripper_pose)
                elif env._action_config.arm == ArmActionMode.DELTA_JOINT_VELOCITY:
                    v_tar = (o_tar.joint_positions - obs.joint_positions) / _DT
                    action.extend(v_tar - obs.joint_velocities)
                    raise TypeError("Warning, delta_joint_velocity is not currently supported")
                elif env._action_config.arm == ArmActionMode.DELTA_JOINT_POSITION:
                    action.extend(o_tar.joint_positions - obs.joint_positions)
                elif env._action_config.arm == ArmActionMode.DELTA_JOINT_TORQUE:
                    action.extend(o_tar.joint_forces - obs.joint_forces)
                    raise TypeError("Warning, delta_joint_torque is not currently supported")
                elif env._action_config.arm == ArmActionMode.DELTA_EE_POSE_WORLD_FRAME:
                    action.extend(o_tar.gripper_pose[:3] - obs.gripper_pose[:3])
                    q = Quaternion(o_tar.gripper_pose[3:7]) * Quaternion(obs.gripper_pose[3:7]).conjugate
                    action.extend(list(q))

                action.append(0.1 if o_tar.gripper_open > 0.9 else -0.1)
                action = np.asarray(action, dtype=np.float32)
                obs_old = obs
                biased_gripper_action = action.copy()
                biased_gripper_action[-1] += 0.5
                obs, reward, done = env.task.step(biased_gripper_action)
                s_array.append(obs_old.get_low_dim_data())
                a_array.append(action)
                r_array.append(reward)
                if done:
                    break
            if input(f"[{len(demo_sub)}/{n_repeat}] r={sum(r_array)} Keep this trajectory ? [y/n]>>>") != 'y':
                continue
            s_array = torch.as_tensor(s_array, dtype=torch.float32)
            a_array = torch.as_tensor(a_array, dtype=torch.float32)
            r_array = torch.as_tensor(r_array, dtype=torch.float32).unsqueeze(dim=-1)
            demo_sub.append((s_array, a_array, r_array))
        demos.extend(demo_sub)
    return demos


def display_all():
    from inspect import getmembers, isclass

    env = None
    for o in getmembers(tasks):
        if isclass(o[1]):
            print(o[0])
            if env is None:
                env = RLBenchEnv(o[0]).init(display=True)
            else:
                env._task_name = o[0]
                env._task = env.env.get_task(o[1])
            env.gen_demo(random=True)


if __name__ == '__main__':
    n_demo = 10
    env_name = "CloseMicrowave2"
    if len(sys.argv) > 1:
        env_name = sys.argv[1]
    if len(sys.argv) > 2:
        n_demo = int(sys.argv[2])
    demos = manuel_demo2(env_name, n_demo=n_demo)
    state_filter = StateFilter(enable=True)
    ss = np.concatenate([s.numpy() for s, a, r in demos], axis=0)
    for s in ss:
        state_filter(s)
    torch.save((demos, state_filter.state_dict()), f"./{env_name}_sample.torch")
    print(f"Demo pack saved in ./{env_name}_sample.torch")
