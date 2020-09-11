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

class SimpleRhythmEnvironment(py_environment.PyEnvironment):
    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minumum=0, name='observation')
        self._state = 0
        self._episode_ended = False
    
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
        
        if action == 1:
            self._episode_ended = True
        else if action == 0:
            self.state += 1
        else:
            raise ValueError('`action` should be 0 or 1.')
        
        if self._episode_ended or self._state >= 1000:
            reward = -np.absolute(500 - self._state) if self._state <= 1000 else -1000
            return ts.termination(np.array([self._state], dtype=np.int32), reward)
        else:
            return ts.transition(
                np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)
    