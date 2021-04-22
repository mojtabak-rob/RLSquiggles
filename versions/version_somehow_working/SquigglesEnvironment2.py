from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from env.Squiggles import Squiggles

from tf_agents.environments import py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

class SquigglesEnvironment(py_environment.PyEnvironment):
    def __init__(self, num_squiggles = 2, num_notes_out=2):
        super(SquigglesEnvironment, self).__init__()
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(num_notes_out,), dtype=np.int32, minimum=0, name='observation')
        self._state = 0
        self._states_until_termination = 1000
        self._episode_ended = False
        self._time_between_squiggles_beats = np.random.randint(10,20)
        self._squiggles_list = [Squiggles() for i in range(num_squiggles)]
        self._squiggles_input = [0 for i in range(16)]
        self.observation = np.zeros(2).astype(np.int32)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = 0
        self._episode_ended = False
        self._time_between_squiggles_beats = np.random.randint(10,20)
        self._squiggles_list = [Squiggles() for i in range(len(self._squiggles_list))]
        self._squiggles_input = [0 for i in range(16)]
        self.observation = np.zeros(2).astype(np.int32)
        self.run_squigs(120*2)

        return ts.restart(self.observation)

    def run_squigs(self, ITER):
        for time in range(ITER):
            time = time*16*self._time_between_squiggles_beats # time is a mock state counter

            for i in range(len(self._squiggles_list)):
                self._squiggles_list[i].update_o()
                if i+1 < len(self._squiggles_list):
                    self._squiggles_list[i+1].update_h(self._squiggles_list[i].o)
                else:
                    self._squiggles_list[0].update_h(self._squiggles_list[i].o)

                    # Take care of observation here
                    for j in range(16,0, -1):
                        if self._squiggles_list[i].hist[j-1] == 1:
                            self.observation[1:] = self.observation[:-1]
                            self.observation[0] = (16 - j)*self._time_between_squiggles_beats
        assert not(np.all(self._squiggles_list[0].hist == 0) and np.all(self._squiggles_list[1].hist == 0))


    def _step(self, action):
        if self._episode_ended:
            return self._reset()

        reward = 0

        if action == 1 or action == 0:
            self._state += 1
            self.observation += 1
        else:
            raise ValueError('`action` should be 0 or 1.')

        if action == 0:
            if self.observation[0] == 0:
                reward = -20
            else:
                reward = 10/self._time_between_squiggles_beats
        elif action == 1:
            if self.observation[0] == 0:
                reward = 20
            elif self.observation[1] - self.observation[0] == self.observation[0]:
                reward = 10
            else:
                reward = -10/self._time_between_squiggles_beats

            # Clicking agent action to discrete squiggle hearing
            current_closeness_to_real_beat = self._state%self._time_between_squiggles_beats
            closest_beat = self._state-current_closeness_to_real_beat
            current_closeness_to_real_beat -= self._time_between_squiggles_beats

            current_i = int((closest_beat/self._time_between_squiggles_beats)%16)
            self._squiggles_input[current_i] = 1

        # Is it time for squiggles to get update?
        if self._state%(self._time_between_squiggles_beats*16) == 0:
            self._squiggles_list[0].update_h(self._squiggles_input)
            for i in range(1, len(self._squiggles_list)):
                self._squiggles_list[i-1].update_o()
                out_in = self._squiggles_list[i-1].o
                self._squiggles_list[i].update_h(out_in)
            self._squiggles_list[-1].update_o()
            #print(self._squiggles_input)
            self._squiggles_input = [0 for i in range(16)]

        play = False
        if self._state%self._time_between_squiggles_beats == 0 and self._squiggles_list[-1].o[int((self._state/self._time_between_squiggles_beats)%16)] == 1:
            play = True

        # Is it time for squig to play, and did it?
        if self._state%self._time_between_squiggles_beats == 0 and \
            self._squiggles_list[-1].o[int((self._state/self._time_between_squiggles_beats)%16)] == 1:
            self.observation[1:] = self.observation[:-1]
            self.observation[0] = 0

        if self._state >= self._states_until_termination:
            self._episode_ended = True
            return ts.termination(self.observation, reward)

        return ts.transition(
            self.observation, reward=reward, discount=0.9)
