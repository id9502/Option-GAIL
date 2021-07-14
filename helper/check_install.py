#!/usr/bin/env python3

from rlbench.tasks import ReachTarget
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.environment import Environment as RLEnvironment


class RLBenchEnv(object):

    def __init__(self, vision=False):
        _observation_config = ObservationConfig()
        _observation_config.set_all_low_dim(True)
        _observation_config.set_all_high_dim(vision)
        self.env = RLEnvironment(action_mode=ActionMode(ArmActionMode.DELTA_JOINT_VELOCITY),
                                 obs_config=_observation_config,
                                 headless=True)
        self.env.launch()
        self.task = self.env.get_task(ReachTarget)

    def __del__(self):
        del self.task
        if self.env is not None:
            self.env.shutdown()
        del self.env

    def reset(self):
        descriptions, obs = self.task.reset()
        return obs.get_low_dim_data()

    def step(self, a):
        obs, reward, terminate, _ = self.task.step(a)
        return obs.get_low_dim_data(), reward, terminate


def main():
    print("========== Starting ===========")
    env = RLBenchEnv(vision=True)
    print("========== Inited ===========")
    s = env.reset()
    print("========== Reset ===========")
    print(s)


if __name__ == "__main__":
    main()
