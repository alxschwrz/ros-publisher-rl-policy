#!/usr/bin/env python
import collections
import rospy
from std_msgs import Floar32MultiArray
from stable_baselines3.sac.policies import MultiInputPolicy
from forwardkinematics.urdfFks.pandaFk import PandaFk
import numpy as np


MODEL_PATH = "/rl_policy.pkl"
ACTION_GAIN = 0.1

def get_ee_pos(obs: dict) -> np.ndarray:
        fk = PandaFk()
        joint_states = obs['joint_state']['position']
        achieved_goal = fk.fk(joint_states, -1, positionOnly=True)
        return achieved_goal


def flatten_observation(observation_dictonary: dict) -> np.ndarray:
    observation_list = []
    for val in observation_dictonary.values():
        if isinstance(val, np.ndarray):
            observation_list += val.tolist()
        elif isinstance(val, dict):
            observation_list += flatten_observation(val).tolist()
    observation_array = np.array(observation_list)
    return observation_array


def convert_observation(obs: dict, goal: np.ndarray ) -> dict:
    goalEnvObs = collections.OrderedDict()
    goalEnvObs['observation']   = np.array(flatten_obervations(obs), dtype=np.float32)
    goalEnvObs['achieved_goal'] = np.array(get_ee_pos(obs), dtype = np.float32)
    goalEnvObs['desired_goal']  = np.array(goal, dtype = np.float32)
    return goalEnvObs


def talker():
    pub = rospy.Publisher('chatter', Float32MultiArray, queue_size=10)
    rospy.init_node('rl_agent_actions', anonymous=True)
    rate = rospy.Rate(100) #Hz
    model = MultiInputPolicy.load(MODEL_PATH)
    while not rospy.is_shutdown():
        ## listen to observation
        # todo: how do i get information here

        ## convert observation to format {'observation': [1x14], 'desired_goal':[1x3], 'achieved_goal':[1x3]
        obs = convert_observation(obs)

        action, _ = model.predict(obs, deterministic=True)

        action = action * ACTION_GAIN

        published_action = Float32MultiArray()
        published_action.data = action
        rospy.loginfo(published_action)
        pub.publish(published_action)
        rate.sleep()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
