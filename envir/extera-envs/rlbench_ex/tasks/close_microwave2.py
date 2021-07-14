from typing import List, Tuple, Union
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from pyrep.objects.dummy import Dummy
from rlbench.backend.task import Task
from rlbench.backend.conditions import Condition


class JointCondition2(Condition):
    def __init__(self, joint: Joint, position: float, is_greater=True):
        """in radians if revoloute, or meters if prismatic"""
        self._joint = joint
        self._pos = position
        self._is_greater = is_greater

    def condition_met(self):
        met = self._joint.get_joint_position() > self._pos
        if self._is_greater:
            met = not met
        return met, False


def get_rand_deg(deg_lo, deg_hi):
    rad_lo, rad_hi = np.deg2rad(deg_lo), np.deg2rad(deg_hi)
    return np.random.rand() * np.fabs(rad_lo - rad_hi) + np.min((rad_hi, rad_lo))


class CloseMicrowave2(Task):

    def init_task(self) -> None:
        self._microwave_joint = Joint('microwave_door_joint')
        self._last_joint_deg = 0.
        self.register_success_conditions([JointCondition2(self._microwave_joint, np.deg2rad(5.), True)])

    def init_episode(self, index: int) -> List[str]:
        self._microwave_joint.set_joint_position(get_rand_deg(30., 75.))
        self._last_joint_deg = 0.
        return ['close microwave',
                'shut the microwave',
                'close the microwave door',
                'push the microwave door shut']

    def init_reward(self) -> None:
        self._last_joint_deg = np.rad2deg(self._microwave_joint.get_joint_position())
        return None

    def reward(self) -> Union[float, None]:
        if self.success()[0]:
            return 200.
        deg = np.rad2deg(self._microwave_joint.get_joint_position())
        r = self._last_joint_deg - deg
        self._last_joint_deg = deg
        return r

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, -3.14 / 4.], [0, 0, 3.14 / 4.]

    def boundary_root(self) -> Object:
        return Shape('boundary_root')

    def is_static_workspace(self) -> bool:
        return True
