#!/usr/bin/env python3
import collections
import rospy
import rospkg
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from stable_baselines3.sac.policies import MultiInputPolicy
from forwardkinematics.urdfFks.pandaFk import PandaFk
import numpy as np


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
        self.rate = rospy.Rate(100) # Hz
        self._model_path = "rl_policy.pkl"
        self._action_gain = 0.1
        self._fk = PandaFk()
        self._published_action = Float64MultiArray()
        self._joint_states = {'joint_states':{'position':np.zeros(7), 'velocity':np.zeros(7)}}
        self.load_model()
        self.action_publisher = rospy.Publisher('panda_joint_velocity_controller/command', Float64MultiArray, queue_size=10)
        self._joint_state_subscriber = rospy.Subscriber('/joint_states', JointState, self.subscriber_callback)


    def run(self):
        while not rospy.is_shutdown():
            ## todo: load goal from controller
            goal = np.array([1.0, 0.2, 0.5])
            obs = self.convert_observation(self._joint_states, goal)
            action, _ = self._model.predict(obs, deterministic=True)
            action = action * self._action_gain
            self._published_action.data = action
            self.action_publisher.publish(self._published_action)


    def load_model(self) -> None:
        rospack = rospkg.RosPack()
        absolute_model_path = rospack.get_path('ros-publisher-rl-policy') + "/scripts/" + self._model_path
        self._model = MultiInputPolicy.load(absolute_model_path)

    def get_ee_pos(self, obs) -> np.ndarray:
        achieved_goal = self._fk.fk(self._joint_states['joint_states']['position'], -1, positionOnly=True)
        return achieved_goal


    def convert_observation(self, obs: dict, goal: np.ndarray) -> dict:
        goalEnvObs = collections.OrderedDict()
        goalEnvObs['observation'] = np.array(flatten_observation(obs), dtype=np.float32)
        goalEnvObs['achieved_goal'] = np.array(self.get_ee_pos(obs), dtype=np.float32)
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

