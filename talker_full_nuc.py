#!/usr/bin/env python3
import collections
import rospy
import rospkg
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from stable_baselines3.sac.policies import MultiInputPolicy
from forwardkinematics.urdfFks.pandaFk import PandaFk
import numpy as np
import yaml
from yaml.loader import SafeLoader


def flatten_observation(observation_dictionary: dict) -> np.ndarray:
    observation_list = []
    for val in observation_dictionary.values():
        if isinstance(val, np.ndarray):
            observation_list += val.tolist()
        elif isinstance(val, dict):
            observation_list += flatten_observation(val).tolist()
    observation_array = np.array(observation_list)
    return observation_array


class RosActionCommunicator:

    def __init__(self):
        rospy.init_node('rl_agent_actions', anonymous=True)
        self._dt = 0.01
        self.rate = rospy.Rate(1/self._dt) # Hz
        self._model_path = "mt_policy_3m_ros.pkl"
        self._joint_limits_path = "joint_limits.yaml"
        self._action_gain = 0.02
        self._fk = PandaFk()
        self._published_action = Float64MultiArray()
        self._joint_states = {'joint_states':{'position':np.zeros(7), 'velocity':np.zeros(7)}}




        ## pick and place
        self._goal = {'position': np.array([0.5, -0.25, 0.35]), 'orientation': np.array([0.0, 0.0, -1.0]), 'task':0}
        ######self._goal = {'position': np.array([0.5, -0.3, 0.5]), 'orientation': np.array([0.5, 0.0, -0.5]), 'task':0}
        self._goal = {'position': np.array([0.5, 0.25, 0.5]), 'orientation': np.array([0.0, 0.0, -1.0]), 'task':0}


        ## reaching
        #self._goal = {'position': np.array([0.3, -0.3, 0.3]), 'orientation': np.array([0.0, 0.0, 0.0]), 'task':1}
        #self._goal = {'position': np.array([0.3, -0.3, 0.6]), 'orientation': np.array([0.0, 0.0, 0.0]), 'task':1}
        #self._goal = {'position': np.array([0.3, 0.3, 0.6]), 'orientation': np.array([0.0, 0.0, 0.0]), 'task':1}
        #self._goal = {'position': np.array([0.3, 0.3, 0.3]), 'orientation': np.array([0.0, 0.0, 0.0]), 'task':1}
        #self._goal = {'position': np.array([0.6, 0.0, 0.4]), 'orientation': np.array([0.0, 0.0, 0.0]), 'task':1}



        ## pointing
        self._goal = {'position': np.array([0.0, 0.0, 0.0]), 'orientation': np.array([1.0, 0.0, 0.0]), 'task':2}
        #self._goal = {'position': np.array([0.0, 0.0, 0.0]), 'orientation': np.array([0.5, 0.5, 0.0]), 'task':2}
        #self._goal = {'position': np.array([0.0, 0.0, 0.0]), 'orientation': np.array([0.5, -0.5, 0.0]), 'task':2}
        #self._goal = {'position': np.array([0.0, 0.0, 0.0]), 'orientation': np.array([0.0, 0.0, -1.0]), 'task':2}
        #self._goal = {'position': np.array([0.0, 0.0, 0.0]), 'orientation': np.array([1.0, 0.0, 0.0]), 'task':2}
        self._joint_limits = {'position': np.zeros([2,7]), 'velocity':np.zeros([2,7])}
        self.load_joint_limits()
        self.load_model()
        self.action_publisher = rospy.Publisher('panda_joint_velocity_controller/command', Float64MultiArray, queue_size=10)
        self._joint_state_subscriber = rospy.Subscriber('/joint_states', JointState, self.subscriber_callback)


    def run(self):
        while not rospy.is_shutdown():
            ## todo: load goal from controller
            #self._goal = {'position': np.array([0.0, 0.0, 0.0]), 'orientation': np.array([1.0, 0.0, 0.0]), 'task':2}
            goal = np.append(self._goal['position'], self._goal['orientation'])
            obs = self.convert_observation(self._joint_states, goal)
            dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'], axis=-1)
            print(dist)
            action, _ = self._model.predict(obs, deterministic=True)
            action = action * self._action_gain
            action = self.clip_action(action)
            self._published_action.data = action
            self.action_publisher.publish(self._published_action)


    def load_model(self) -> None:
        rospack = rospkg.RosPack()
        absolute_model_path = rospack.get_path('ros-publisher-rl-policy') + "/scripts/" + self._model_path
        self._model = MultiInputPolicy.load(absolute_model_path)


    def load_joint_limits(self) -> None:
        rospack = rospkg.RosPack()
        absolute_joint_limit_path = rospack.get_path('ros-publisher-rl-policy') + "/scripts/" + self._joint_limits_path
        with open(absolute_joint_limit_path, 'r') as f:
            config_limits = yaml.load(f, Loader=SafeLoader)
        self._joint_limits['position'] = config_limits['position']
        self._joint_limits['velocity'] = config_limits['velocity']


    def get_ee_pos(self, obs) -> np.ndarray:
        achieved_ee_pos = self._fk.fk(self._joint_states['joint_states']['position'], -1, positionOnly=True)
        return achieved_ee_pos


    def get_ee_dir(self, obs) -> np.ndarray:
        ee_pos = self._fk.fk(self._joint_states['joint_states']['position'], -1, positionOnly=True)
        last_link_pos = self._fk.fk(self._joint_states['joint_states']['position'], -2, positionOnly=True)
        direction = ee_pos - last_link_pos
        normalized_direction = direction / np.sqrt(np.sum(direction ** 2))
        return normalized_direction


    def clip_action(self, action) -> np.ndarray:
        clipped_action = np.zeros(action.shape)
        next_state = self._joint_states['joint_states']['position'] + action * self._dt
        for i, act in enumerate(action):
            if self._joint_limits['position'][0][i] * 1.1  <= next_state[i] <= 0.9 * self._joint_limits['position'][1][i]:
                clipped_action[i] = act
            else:
                print('clipping ', i)
                clipped_action[i] = -act
        return clipped_action


    def convert_observation(self, obs: dict, goal: np.ndarray) -> dict:
        goalEnvObs = collections.OrderedDict()
        observation = np.append(flatten_observation(obs), self._goal['task']) ## append taskID obsSpace 15x1
        goalEnvObs['observation'] = np.array(observation, dtype=np.float32)

        ## reaching
        if self._goal['task'] == 1:
                achieved_goal = np.append(self.get_ee_pos(obs), np.zeros(3))

        ## pointing
        if self._goal['task'] == 2:
                achieved_goal = np.append(np.zeros(3), self.get_ee_dir(obs))

        ## reaching + pointing
        if self._goal['task'] == 0:
                achieved_goal = np.append(self.get_ee_pos(obs), self.get_ee_dir(obs))

        goalEnvObs['achieved_goal'] = np.array(achieved_goal, dtype=np.float32)

        goalEnvObs['desired_goal'] = np.array(goal, dtype=np.float32)
        return goalEnvObs


    def subscriber_callback(self, data):
        self._joint_states['joint_states']['position'] = np.array(data.position[3:10])
        self._joint_states['joint_states']['velocity'] = np.array(data.velocity[3:10])


if __name__ == '__main__':
    try:
        communicator = RosActionCommunicator()
        communicator.run()
    except rospy.ROSInterruptException:
        pass

