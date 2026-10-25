import os
import numpy as np

from gymnasium.utils.ezpickle import EzPickle
from gymnasium.spaces import Box, Discrete
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv
from gymnasium_robotics.utils import rotations

MODEL_XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fetch", "mo_reach.xml")


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class BaseFetchEnv(MujocoRobotEnv):
    """Superclass for all Fetch environments."""

    def __init__(
        self,
        gripper_extra_height,
        block_gripper,
        has_object: bool,
        target_in_the_air,
        target_offset,
        obj_range,
        target_range,
        distance_threshold,
        reward_type,
        **kwargs
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """

        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type

        super().__init__(n_actions=4, **kwargs)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _set_action(self, action):
        assert action.shape == (4,)
        action = (
            action.copy()
        )  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [
            1.0,
            0.0,
            1.0,
            0.0,
        ]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        return action

    def _get_obs(self):
        (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
        ) = self.generate_mujoco_observations()

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())

        obs = np.concatenate(
            [
                grip_pos,
                object_pos.ravel()[:2],
                object_rel_pos.ravel(),
                gripper_state,
                object_rot.ravel(),
                object_velp.ravel(),
                object_velr.ravel(),
                grip_velp[:2],
                gripper_vel,
            ]
        )

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    def generate_mujoco_observations(self):

        raise NotImplementedError

    def _get_gripper_xpos(self):

        raise NotImplementedError

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                -self.target_range, self.target_range, size=3
            )
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                -self.target_range, self.target_range, size=3
            )
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)


DEFAULT_CAMERA_CONFIG = {
    "distance": 2.5,
    "azimuth": 132.0,
    "elevation": -14.0,
    "lookat": np.array([1.3, 0.75, 0.55]),
}

class MujocoFetchEnv(BaseFetchEnv):
    def __init__(self, default_camera_config: dict = DEFAULT_CAMERA_CONFIG, **kwargs):
        super().__init__(default_camera_config=default_camera_config, **kwargs)

        goal = self.initial_gripper_xpos[:3]
        goal += self.target_offset
        goal[2] = self.target_range + 0.5

        g0 = goal.copy()
        g0[0] += self.target_range +0.15

        g1 = goal.copy()
        g1[0] += -self.target_range -0.15

        g2 = goal.copy()
        g2[1] += self.target_range +0.15

        g3 = goal.copy()
        g3[1] += -self.target_range -0.15

        g4 = goal.copy()
        g4[0] += self.target_range +0.1
        g4[1] += self.target_range +0.1

        g5 = goal.copy()
        g5[0] += -self.target_range -0.1
        g5[1] += -self.target_range -0.1

        g6 = goal.copy()
        g6[0] += self.target_range +0.1
        g6[1] += -self.target_range -0.1

        g7 = goal.copy()
        g7[0] += -self.target_range -0.1
        g7[1] += self.target_range +0.1

        self.targets = [g0, g1, g2, g3, g4, g5, g6, g7]

    def _step_callback(self):
        if self.block_gripper:
            self._utils.set_joint_qpos(
                self.model, self.data, "robot0:l_gripper_finger_joint", 0.0
            )
            self._utils.set_joint_qpos(
                self.model, self.data, "robot0:r_gripper_finger_joint", 0.0
            )
            self._mujoco.mj_forward(self.model, self.data)

    def _set_action(self, action):
        action = super()._set_action(action)

        # Apply action to simulation.
        self._utils.ctrl_set_action(self.model, self.data, action)
        self._utils.mocap_set_action(self.model, self.data, action)

    def generate_mujoco_observations(self):
        # positions
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")

        dt = self.n_substeps * self.model.opt.timestep
        grip_velp = (
            self._utils.get_site_xvelp(self.model, self.data, "robot0:grip") * dt
        )

        robot_qpos, robot_qvel = self._utils.robot_get_obs(
            self.model, self.data, self._model_names.joint_names
        )
        if self.has_object:
            object_pos = self._utils.get_site_xpos(self.model, self.data, "object0")
            # rotations
            object_rot = rotations.mat2euler(
                self._utils.get_site_xmat(self.model, self.data, "object0")
            )
            # velocities
            object_velp = (
                self._utils.get_site_xvelp(self.model, self.data, "object0") * dt
            )
            object_velr = (
                self._utils.get_site_xvelr(self.model, self.data, "object0") * dt
            )
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = (
                object_rot
            ) = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]

        gripper_vel = (
            robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric

        return (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
        )

    def _get_gripper_xpos(self):
        body_id = self._model_names.body_name2id["robot0:gripper_link"]
        return self.data.xpos[body_id]

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        for i in range(8):
            site_id = self._mujoco.mj_name2id(
                self.model, self._mujoco.mjtObj.mjOBJ_SITE, f"target{i}"
            )
            self.model.site_pos[site_id] = self.targets[i] - sites_offset[0]
        self._mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            """ while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2]  + self.np_random.uniform(
                    -self.obj_range, self.obj_range, size=2
                ) """
            object_qpos = self._utils.get_joint_qpos(
                self.model, self.data, "object0:joint"
            )
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self._utils.set_joint_qpos(
                self.model, self.data, "object0:joint", object_qpos
            )

        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        self._utils.reset_mocap_welds(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)

        # Move end effector into position.
        gripper_target = np.array(
            [-0.498, 0.005, -0.431 + self.gripper_extra_height]
        ) + self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self._utils.set_mocap_pos(self.model, self.data, "robot0:mocap", gripper_target)
        self._utils.set_mocap_quat(
            self.model, self.data, "robot0:mocap", gripper_rotation
        )
        for _ in range(10):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        # Extract information for sampling goals.
        self.initial_gripper_xpos = self._utils.get_site_xpos(
            self.model, self.data, "robot0:grip"
        ).copy()
        if self.has_object:
            self.height_offset = self._utils.get_site_xpos(
                self.model, self.data, "object0"
            )[2]



class MujocoMOFetchReachEnv(MujocoFetchEnv, EzPickle):
    """
    ## Description

    This environment was introduced in ["Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research"](https://arxiv.org/abs/1802.09464).

    The task in the environment is for a manipulator to move the end effector to a randomly selected position in the robot's workspace. The robot is a 7-DoF [Fetch Mobile Manipulator](https://fetchrobotics.com/) with a two-fingered parallel gripper.
    The robot is controlled by small displacements of the gripper in Cartesian coordinates and the inverse kinematics are computed internally by the MuJoCo framework. The task is also continuing which means that the robot has to maintain the end effector's
    position for an indefinite period of time.

    The control frequency of the robot is of `f = 25 Hz`. This is achieved by applying the same action in 20 subsequent simulator step (with a time step of `dt = 0.002 s`) before returning the control to the robot.

    ## Action Space

    The action space is a `Box(-1.0, 1.0, (4,), float32)`. An action represents the Cartesian displacement dx, dy, and dz of the end effector. In addition to a last action that controls closing and opening of the gripper. This last action is not required since
    there is no object to be manipulated, thus its value won't generate any control output.

    | Num | Action                                                 | Control Min | Control Max | Name (in corresponding XML file)                                | Joint | Unit         |
    | --- | ------------------------------------------------------ | ----------- | ----------- | --------------------------------------------------------------- | ----- | ------------ |
    | 0   | Displacement of the end effector in the x direction dx | -1          | 1           | robot0:mocap                                                    | hinge | position (m) |
    | 1   | Displacement of the end effector in the y direction dy | -1          | 1           | robot0:mocap                                                    | hinge | position (m) |
    | 2   | Displacement of the end effector in the z direction dz | -1          | 1           | robot0:mocap                                                    | hinge | position (m) |
    | 3   | -                                                      | -1          | 1           | -                                                               | -     | -            |

    ## Observation Space

    The observation is a `goal-aware observation space`. It consists of a dictionary with information about the robot's end effector state and goal. The kinematics observations are derived from Mujoco bodies known as [sites](https://mujoco.readthedocs.io/en/latest/XMLreference.html?highlight=site#body-site)
    attached to the body of interest, the end effector. Also to take into account the temporal influence of the step time, velocity values are multiplied by the step time dt=number_of_sub_steps*sub_step_time. The dictionary consists of the following 3 keys:

    * `observation`: its value is an `ndarray` of shape `(10,)`. It consists of kinematic information of the end effector. The elements of the array correspond to the following:

    | Num | Observation                                                                                                                           | Min    | Max    | Site Name (in corresponding XML file) | Joint Name (in corresponding XML file) |Joint Type| Unit                     |
    |-----|---------------------------------------------------------------------------------------------------------------------------------------|--------|--------|---------------------------------------|----------------------------------------|----------|--------------------------|
    | 0   | End effector x position in global coordinates                                                                                         | -Inf   | Inf    | robot0:grip                           |-                                       |-         | position (m)             |
    | 1   | End effector y position in global coordinates                                                                                         | -Inf   | Inf    | robot0:grip                           |-                                       |-         | position (m)             |
    | 2   | End effector z position in global coordinates                                                                                         | -Inf   | Inf    | robot0:grip                           |-                                       |-         | position (m)             |
    | 3   | Joint displacement of the right gripper finger                                                                                        | -Inf   | Inf    |-                                      | robot0:r_gripper_finger_joint          | hinge    | position (m)             |
    | 4   | Joint displacement of the left gripper finger                                                                                         | -Inf   | Inf    |-                                      | robot0:l_gripper_finger_joint          | hinge    | position (m)             |
    | 5   | End effector linear velocity x direction                                                                                              | -Inf   | Inf    | robot0:grip                           |-                                       |-         | velocity (m/s)           |
    | 6   | End effector linear velocity y direction                                                                                              | -Inf   | Inf    | robot0:grip                           |-                                       |-         | velocity (m/s)           |
    | 7   | End effector linear velocity z direction                                                                                              | -Inf   | Inf    | robot0:grip                           |-                                       |-         | velocity (m/s)           |
    | 8   | Right gripper finger linear velocity                                                                                                  | -Inf   | Inf    |-                                      | robot0:r_gripper_finger_joint          | hinge    | velocity (m/s)           |
    | 9   | Left gripper finger linear velocity                                                                                                   | -Inf   | Inf    |-                                      | robot0:l_gripper_finger_joint          | hinge    | velocity (m/s)           |

    * `desired_goal`: this key represents the final goal to be achieved. In this environment it is a 3-dimensional `ndarray`, `(3,)`, that consists of the three cartesian coordinates of the desired final end effector position `[x,y,z]`. The elements of the array are the following:

    | Num | Observation                                                                                                                           | Min    | Max    | Site Name (in corresponding XML file) |Unit          |
    |-----|---------------------------------------------------------------------------------------------------------------------------------------|--------|--------|---------------------------------------|--------------|
    | 0   | Final goal end effector position in the x coordinate                                                                                  | -Inf   | Inf    | robot0:grip                           | position (m) |
    | 1   | Final goal end effector position in the y coordinate                                                                                  | -Inf   | Inf    | robot0:grip                           | position (m) |
    | 2   | Final goal end effector position in the z coordinate                                                                                  | -Inf   | Inf    | robot0:grip                           | position (m) |

    * `achieved_goal`: this key represents the current state of the end effector, as if it would have achieved a goal. This is useful for goal orientated learning algorithms such as those that use [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) (HER).
    The value is an `ndarray` with shape `(3,)`. The elements of the array are the following:

    | Num | Observation                                                                                                                           | Min    | Max    | Site Name (in corresponding XML file) |Unit          |
    |-----|---------------------------------------------------------------------------------------------------------------------------------------|--------|--------|---------------------------------------|--------------|
    | 0   | Current end effector position in the x coordinate                                                                                     | -Inf   | Inf    | robot0:grip                           | position (m) |
    | 1   | Current end effector position in the y coordinate                                                                                     | -Inf   | Inf    | robot0:grip                           | position (m) |
    | 2   | Current end effector position in the z coordinate                                                                                     | -Inf   | Inf    | robot0:grip                           | position (m) |


    ## Rewards

    The reward can be initialized as `sparse` or `dense`:
    - *sparse*: the returned reward can have two values: `-1` if the end effector hasn't reached its final target position, and `0` if the end effector is in the final target position (the robot is considered to have reached the goal if the Euclidean distance between
    the end effector and the goal is lower than 0.05 m).
    - *dense*: the returned reward is the negative Euclidean distance between the achieved goal position and the desired goal.

    To initialize this environment with one of the mentioned reward functions the type of reward must be specified in the id string when the environment is initialized. For `sparse` reward the id is the default of the environment, `FetchReach-v3`. However, for `dense`
    reward the id must be modified to `FetchReachDense-v3` and initialized as follows:

    ```python
    import gymnasium as gym
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)

    env = gym.make('FetchReachDense-v3')
    ```

    ## Starting State

    When the environment is reset the gripper is placed in the following global cartesian coordinates `(x,y,z) = [1.3419 0.7491 0.555] m`, and its orientation in quaternions is `(w,x,y,z) = [1.0, 0.0, 1.0, 0.0]`. The joint positions are computed by inverse kinematics
    internally by MuJoCo. The base of the robot will always be fixed at `(x,y,z) = [0.405, 0.48, 0]` in global coordinates.

    The gripper's target position is randomly selected by adding an offset to the initial grippers position `(x,y,z)` sampled from a uniform distribution with a range of `[-0.15, 0.15] m`.

    ## Episode End

    The episode will be `truncated` when the duration reaches a total of `max_episode_steps` which by default is set to 50 timesteps.
    The episode is never `terminated` since the task is continuing with infinite horizon.

    ## Arguments

    To increase/decrease the maximum number of timesteps before the episode is `truncated` the `max_episode_steps` argument can be set at initialization. The default value is 50. For example, to increase the total number of timesteps to 100 make the environment as follows:

    ```python
    import gymnasium as gym
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)

    env = gym.make('FetchReach-v3', max_episode_steps=100)
    ```

    ## Version History
    * v3: Fixed bug: `env.reset()` not properly resetting the internal state. Fetch environments now properly reset their state (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium-Robotics/issues/207)).
    * v2: the environment depends on the newest [mujoco python bindings](https://mujoco.readthedocs.io/en/latest/python.html) maintained by the MuJoCo team in Deepmind.
    * v1: the environment depends on `mujoco_py` which is no longer maintained.
    """

    def __init__(self, reward_type: str = "sparse", d=4, **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.4049,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }
        MujocoFetchEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=False,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.1,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)

        self.d = d
        self.reward_space = Box(low=-1.0, high=0.0, shape=(self.d,), dtype=np.float32)
        self.reward_dim = self.d


    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        if self.d == 4: 
            r0 = self.compute_reward(obs["achieved_goal"], self.targets[0], None)
            r1 = self.compute_reward(obs["achieved_goal"], self.targets[1], None)
            r2 = self.compute_reward(obs["achieved_goal"], self.targets[2], None)
            r3 = self.compute_reward(obs["achieved_goal"], self.targets[3], None)
            reward = np.array([r0, r1, r2, r3], dtype=np.float32)
        elif self.d == 6:
            r0 = self.compute_reward(obs["achieved_goal"], self.targets[0], None)
            r1 = self.compute_reward(obs["achieved_goal"], self.targets[1], None)
            r2 = self.compute_reward(obs["achieved_goal"], self.targets[2], None)
            r3 = self.compute_reward(obs["achieved_goal"], self.targets[3], None)
            r4 = self.compute_reward(obs["achieved_goal"], self.targets[4], None)
            r5 = self.compute_reward(obs["achieved_goal"], self.targets[5], None)
            reward = np.array([r0, r1, r2, r3, r4, r5], dtype=np.float32)
        elif self.d == 8:
            r0 = self.compute_reward(obs["achieved_goal"], self.targets[0], None)
            r1 = self.compute_reward(obs["achieved_goal"], self.targets[1], None)
            r2 = self.compute_reward(obs["achieved_goal"], self.targets[2], None)
            r3 = self.compute_reward(obs["achieved_goal"], self.targets[3], None)
            r4 = self.compute_reward(obs["achieved_goal"], self.targets[4], None)
            r5 = self.compute_reward(obs["achieved_goal"], self.targets[5], None)    
            r6 = self.compute_reward(obs["achieved_goal"], self.targets[6], None)
            r7 = self.compute_reward(obs["achieved_goal"], self.targets[7], None)
            reward = np.array([r0, r1, r2, r3, r4, r5, r6, r7], dtype=np.float32)
 
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":

    import mo_gymnasium as mo_gym
    import envs.mo_push
    from envs.mo_push.mo_push import FetchDiscreteAction, FetchDiscreteManhattanAction, FetchObservationWrapper

    def make_env(record_video=False):
        env = mo_gym.make("mo-reach-d4-v2", render_mode="human")
        env = FetchDiscreteManhattanAction(env, use_gripper=False)
        env = FetchObservationWrapper(env)
        #if record_video:
        #    env = RecordVideo(env, "videos/reacher/", episode_trigger=lambda e: e % 1 == 0)
        return env
    
    env = make_env()

    env.reset()
    while True:
        action = env.action_space.sample()
        #action = int(input())
        obs, reward, terminated, truncated, info = env.step(action)
        print(obs, reward, terminated, truncated, info)
        env.render()
