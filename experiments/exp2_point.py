import time
import numpy as np
import pandas as pd
from datetime import datetime
from talker_full_nuc import RosActionCommunicator


class Experiment2Pointing(RosActionCommunicator):

        def __init__(self):
                super().__init__()
                self._sequence_duration = 10.0
                self._experiment_duration = 200.0
                self._sequence_id = 0
                self._sequence = "forward"
                self._recorder_file_name = "exp2_point"
        
        def start_timer(self):
                self._start_time = time.time()

        def set_sequence_time_start(self):
                self._start_sequence = time.time()

        def get_relative_time_to_start(self):
                return time.time() - self._start_time

        def get_sequence_duration(self):
                return time.time() - self._start_sequence
        
        def set_experiment_components(self):
                self._experiment_goal_components = {"forward": [1.0, 0.0, 0.0], "left": [0.5, 0.5, 0.0], "right": [0.5, -0.5, 0.0], "down": [0.0, 0.0, -1.0]}

        def update_goal(self):
                self._sequence = list(self._experiment_goal_components.keys())[self._sequence_id]
                self._goal = {'position': np.array([0.0, 0.0, 0.0]), 'orientation': np.array(self._experiment_goal_components[self._sequence]), 'task':2}

        def reset_recorder(self):
                self.data = {'timestep': [],
                'dist2goal': [],
                'achieved_goal': [],
                'rachieved_goal': [],
                'desired_goal': [],
                'joint_states': [],
                }

        
        def record(self, obs):
                self.data['timestep'].append(self.get_relative_time_to_start())
                self.data['dist2goal'].append(np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'], axis=-1))
                self.data['achieved_goal'].append(obs['achieved_goal'])
                self.data['desired_goal'].append(obs['desired_goal'])
                self.data['joint_states'].append(obs['observation'])

        def write_recording(self):
                df = pd.DataFrame.from_dict(self.data)
                dateTime = str(datetime.now()).split(" ")[0]+str(datetime.now()).split(" ")[1]
                df.to_csv(self._recorder_file_name+dateTime+('.csv'))

        
        def run_pointing_experiment(self):
                self.start_timer()
                self.set_sequence_time_start()
                while not rospy.is_shutdown():
                ## todo: load goal from controller
                        if self.get_sequence_duration() > self._sequence_duration:
                                self._sequence_id += 1
                                if self._sequence_id > 3:
                                        self._sequence_id = 0
                                self.update_goal()
                        goal = np.append(self._goal['position'], self._goal['orientation'])
                        obs = self.convert_observation(self._joint_states, goal)
                        action, _ = self._model.predict(obs, deterministic=True)
                        action = action * self._action_gain
                        action = self.clip_action(action)
                        self._published_action.data = action
                        self.action_publisher.publish(self._published_action)
                        if self.get_relative_time_to_start() > self._experiment_duration:
                                self.write_recording()
                                break


if __name__ == '__main__':
    try:
        communicator = Experiment2Pointing()
        communicator.run_pointing_experiment()
    except rospy.ROSInterruptException:
        pass