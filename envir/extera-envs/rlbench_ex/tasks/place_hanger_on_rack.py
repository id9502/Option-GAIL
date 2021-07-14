from typing import List, Union
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.cartesian_path import CartesianPath
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, NothingGrasped
from rlbench.backend.spawn_boundary import SpawnBoundary
import numpy as np


def is_grasped(gripper, obj):
    handle = obj.get_handle()
    return len([ob for ob in gripper.get_grasped_objects() if handle == ob.get_handle()]) > 0


class PlaceHangerOnRack(Task):

    def init_task(self) -> None:
        self.hanger_holder = Shape('hanger_holder')
        self.hanger = Shape('clothes_hanger')
        self.wp0 = Dummy("waypoint0")
        self.wp1 = Dummy("waypoint1")
        self.wp4 = np.asarray(CartesianPath("waypoint4").get_pose_on_path(0.)[0])
        self.gripper_tip = Dummy(f"{self.robot.arm.get_name()}_tip")
        self.success_detector = ProximitySensor('success_detector0')
        self.register_success_conditions([
            NothingGrasped(self.robot.gripper),
            DetectedCondition(self.hanger, self.success_detector)
        ])
        self.register_graspable_objects([self.hanger])
        self.workspace_boundary = SpawnBoundary([Shape('workspace')])
        self.sub_task = 0
        self.last_distance = 0.

    def init_episode(self, index: int) -> List[str]:
        if not self.is_static_workspace():
            self.workspace_boundary.clear()
            self.workspace_boundary.sample(self.hanger_holder)
        return ['pick up the hanger and place in on the rack'
                'put the hanger on the rack',
                'hang the hanger up',
                'grasp the hanger and hang it up']

    def init_reward(self) -> None:
        self.sub_task = 0
        self.last_distance = np.linalg.norm(self.gripper_tip.get_position(self.wp0))
        return None

    def reward(self) -> Union[float, None]:
        if self.sub_task == 0:  # reaching the rack, use distance to waypoint_0 as reward
            d = np.linalg.norm(self.gripper_tip.get_position(self.wp0))
            r = max(self.last_distance - d, -1.) * 10.
            self.last_distance = d
            if np.abs(d) < 0.005:
                r += 10.
                self.sub_task = 1
                self.last_distance = np.linalg.norm(self.gripper_tip.get_position(self.wp1))
            return r
        elif self.sub_task == 1:  # grasping the rack
            d = np.linalg.norm(self.gripper_tip.get_position(self.wp1))
            r = max(self.last_distance - d, -1.) * 10.
            self.last_distance = d
            if is_grasped(self.robot.gripper, self.hanger):
                r += 10.
                self.sub_task = 2
                self.last_distance = np.linalg.norm(self.wp1.get_position() - self.wp4)
            return r
        elif self.sub_task == 2:  # moving the rack
            d = np.linalg.norm(self.wp1.get_position() - self.wp4)
            r = max(self.last_distance - d, -1.) * 10.
            self.last_distance = d
            if self.success_detector.is_detected(self.hanger):
                r += 10.
                self.sub_task = 3
            return r
        elif self.sub_task == 3:  # nothing in hand
            if not is_grasped(self.robot.gripper, self.hanger) and self.success_detector.is_detected(self.hanger):
                self.sub_task = 4
                return 100.
            else:
                return -0.001
        else:
            return 0.

    def variation_count(self) -> int:
        return 1

    def is_static_workspace(self) -> bool:
        return True

    def _feasible(self, waypoints):
        return True, -1


class PlaceHangerOnRack_old(Task):

    def init_task(self) -> None:
        self.hanger_holder = Shape('hanger_holder')
        hanger = Shape('clothes_hanger')
        success_detector = ProximitySensor('success_detector0')
        self.register_success_conditions([
            NothingGrasped(self.robot.gripper),
            DetectedCondition(hanger, success_detector)
        ])
        self.register_graspable_objects([hanger])
        self.workspace_boundary = SpawnBoundary([Shape('workspace')])

    def init_episode(self, index: int) -> List[str]:
        self.workspace_boundary.clear()
        self.workspace_boundary.sample(self.hanger_holder)
        return ['pick up the hanger and place in on the rack'
                'put the hanger on the rack',
                'hang the hanger up',
                'grasp the hanger and hang it up']

    def variation_count(self) -> int:
        return 1

    def is_static_workspace(self) -> bool:
        return True
