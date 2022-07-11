#!/usr/bin/env python
import collections
import rospy
from std_msgs.msg import Float32MultiArray
from stable_baselines3.sac.policies import MultiInputPolicy
from forwardkinematics.urdfFks.pandaFk import PandaFk
import numpy as np

MODEL_PATH = "/rl_policy.pkl"
ACTION_GAIN = 0.1


def flatten_observation(observation_dictonary: dict) -> np.ndarray:
    observation_list = []
    for val in observation_dictonary.values():
        if isinstance(val, np.ndarray):
            observation_list += val.tolist()
        elif isinstance(val, dict):
            observation_list += flatten_observation(val).tolist()
    observation_array = np.array(observation_list)
    return observation_array


class RosActionCommunicator:

    def __init__(self):
        self._model_path = "/rl_policy.pkl"
        self._action_gain = 0.1
        self._fk = PandaFk()
        self._published_action = Float32MultiArray()
        self.rate = rospy.Rate(100)
        self.load_model()

    def load_model(self) -> None:
        self._model = MultiInputPolicy.load(MODEL_PATH)

    def get_ee_pos(self, obs) -> np.ndarray:
        joint_states = obs['joint_state']['position']
        achieved_goal = self._fk.fk(joint_states, -1, positionOnly=True)
        return achieved_goal

    def convert_observation(self, obs: dict, goal: np.ndarray) -> dict:
        goalEnvObs = collections.OrderedDict()
        goalEnvObs['observation'] = np.array(flatten_observation(obs), dtype=np.float32)
        goalEnvObs['achieved_goal'] = np.array(self.get_ee_pos(obs), dtype=np.float32)
        goalEnvObs['desired_goal'] = np.array(goal, dtype=np.float32)
        return goalEnvObs

    def talker(self):
        pub = rospy.Publisher('chatter', Float32MultiArray, queue_size=10)
        rospy.init_node('rl_agent_actions', anonymous=True)
        while not rospy.is_shutdown():
            # todo: how to listen to observation information?
            self.subscribe_joint_states()
            obs = dict()
            goal = np.zeros(3)

            obs = self.convert_observation(obs, goal)

            action, _ = self._model.predict(obs, deterministic=True)

            action = action * self._action_gain

            self._published_action.data = action
            rospy.loginfo(self._published_action)
            pub.publish(self._published_action)
            self.rate.sleep()


    def subscriber_callback(self, data):
        rospy.loginfo("Joint states: {}", data.data)
        ## get goal and joint_state information and store where?


    def subscribe_joint_states(self):
        rospy.init_node('joint_state_listener', anonymous=True)

        rospy.Subscriber('chatter_XY', Float32MultiArray, self.subscriber_callback) # todo: what exactly is chatter

        rospy.spin()


if __name__ == '__main__':
    try:
        communicator = RosActionCommunicator()
        communicator.talker()
    except rospy.ROSInterruptException:
        pass
