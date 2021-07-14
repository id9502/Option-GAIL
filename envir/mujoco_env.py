import os
import torch
try:
    import pybullet_envs
except ImportError:
    print("Warning: pybullet not installed, bullet environments will be unavailable")
import gym


class MujocoEnv(object):
    def __init__(self, task_name: str = "HalfCheetah-v2"):
        self.task_name = task_name
        self.env = None
        self.display = False

    def init(self, display=False):
        self.env = gym.make(self.task_name)
        self.display = display
        return self

    def reset(self, random: bool = False):
        s = self.env.reset()
        return s

    def step(self, a):
        s, reward, terminate, info = self.env.step(a)
        if self.display:
            self.env.render()
        return s, reward, terminate

    def state_action_size(self):
        if self.env is not None:
            s_dim = self.env.observation_space.shape[0]
            a_dim = self.env.action_space.shape[0]
        else:
            env = gym.make(self.task_name)
            s_dim = env.observation_space.shape[0]
            a_dim = env.action_space.shape[0]
            env.close()
        return s_dim, a_dim


def get_demo(path="", n_demo=2048, display=False):
    if os.path.isfile(path):
        print(f"Demo Loaded from {path}")
        samples, filter_state = torch.load(path)
        n_current_demo = 0
        sample = []
        for traj in samples:
            sample.append(traj)
            n_current_demo += traj[2].size(0)
            if n_current_demo >= n_demo:
                break
        if n_current_demo < n_demo:
            print(f"Warning, demo package contains less demo than required ({n_current_demo}/{n_demo})")
        return sample, filter_state
    else:
        from model.option_policy import Policy
        from utils.state_filter import StateFilter
        from default_config import mujoco_config

        expert_path = f"./data/ppo-expert/{mujoco_config.env_name}_expert.torch"
        config_path = f"./data/ppo-expert/{mujoco_config.env_name}_expert.log"
        if os.path.isfile(config_path):
            mujoco_config.load_saved(config_path)

        n_demo = 409600

        mujoco_config.device = "cpu"
        use_rs = mujoco_config.use_state_filter

        env = MujocoEnv(mujoco_config.env_name)
        dim_s, dim_a = env.state_action_size()
        env.init(display=display)

        policy_state, filter_state = torch.load(expert_path, map_location="cpu")
        policy = Policy(mujoco_config, dim_s, dim_a)
        policy.load_state_dict(policy_state)
        rs = StateFilter(enable=use_rs)
        rs.load_state_dict(filter_state)

        sample = []
        n_current_demo = 0
        while n_current_demo < n_demo:
            with torch.no_grad():
                s_array = []
                a_array = []
                r_array = []
                s, done = env.reset(), False
                while not done:
                    st = torch.as_tensor(s, dtype=torch.float32).unsqueeze(dim=0)
                    s_array.append(st.clone())
                    at = policy.sample_action(rs(st, fixed=True), fixed=True)
                    a_array.append(at.clone())
                    s, r, done = env.step(at.squeeze(dim=0).numpy())
                    r_array.append(r)
                a_array = torch.cat(a_array, dim=0)
                s_array = torch.cat(s_array, dim=0)
                r_array = torch.as_tensor(r_array, dtype=torch.float32).unsqueeze(dim=1)
                print(f"R-Sum={r_array.sum()}, L={r_array.size(0)}")
                keep = input(f"{n_current_demo}/{n_demo} Keep this ? [y|n]>>>")
                if keep == 'y':
                    sample.append((s_array, a_array, r_array))
                    n_current_demo += r_array.size(0)
        torch.save((sample, rs.state_dict()), path)
    return sample, filter_state


if __name__ == "__main__":
    get_demo(display=True)
