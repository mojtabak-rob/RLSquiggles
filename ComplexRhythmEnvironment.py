from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

class ComplexRhythmEnvironment(py_environment.PyEnvironment):
    def __init__(self, bpm = 120, offset = 0):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.int32, minimum=0, name='observation')
        self._state = 0
        self._episode_ended = False
        self._time_between_beats = 60*1000//bpm
        self._offset = offset
        self._time_since_last_play = 0
        self._number_of_plays = 0
        self._number_of_real_plays = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = 0
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=np.int32))

    def _step(self, action):
        if self._episode_ended:
            return self._reset()

        if action == 1 or action == 0:
            self._state += 1
        else:
            raise ValueError('`action` should be 0 or 1.')

        output = 1 if (self._state-offset)%self._time_between_beats == 0 else 0
        self._number_of_real_plays += output

        if self._state >= 10000:
            #50 is a random constant here, should probably be tweaked
            reward = -np.absolute(self._number_of_plays - self._number_of_real_plays)*50
            return ts.termination(np.array([self._state, output], dtype=np.int32), reward)
        else:
            reward = 0

            if action == 1:
                self._number_of_plays += 1
                if _time_since_last_play < self._time_between_beats/3:
                    reward = -100 #Random number, probably needs tweaking
                else:
                    current_closeness_to_real_beat = (self._state-self._offset)%self._time_between_beats
                    if current_closeness_to_real_beat > self._time_between_beats/2:
                        current_closeness_to_real_beat -= self._time_between_beats
                    reward = -np.absolute(0 - current_closeness_to_real_beat)

                self._time_since_last_play = 0

            return ts.transition(
                np.array([self._state, output], dtype=np.int32), reward=reward, discount=1.0)
