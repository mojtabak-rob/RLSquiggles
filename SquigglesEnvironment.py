from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from Squiggles import Squiggles

from tf_agents.environments import py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

class SquigglesEnvironment(py_environment.PyEnvironment):
    def __init__(self, bpm = 120, num_squiggles = 2, num_notes_out=2):
        super(SquigglesEnvironment, self).__init__()
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(num_notes_out+1,), dtype=np.int32, minimum=0, name='observation')
        self._state = 0
        self._time_since_real_plays = np.array([0 for _ in range(num_notes_out+1)]).astype(np.int32)
        self._episode_ended = False
        self._time_between_squiggles_beats = 60*100//(4*bpm)
        self._time_since_last_play = 0
        self._number_of_plays = 0
        self._number_of_real_plays = 0
        self._squiggles_list = [Squiggles() for i in range(num_squiggles)]
        self._squiggles_input = [0 for i in range(16)]
        for squig in self._squiggles_list:
            squig.update_o()

        notes_filled = 1
        for i in range(1,17):
            output = self._squiggles_list[-1].o
            if output[-i] == 1:
                self._time_since_real_plays[-notes_filled] = self._time_between_squiggles_beats*i
                notes_filled+=1
                if notes_filled > num_notes_out:
                    break

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = 0
        self._episode_ended = False
        #self._time_since_real_plays = np.zeros(self._time_since_real_plays.shape).astype(np.int32)
        self._number_of_plays = 0
        #self._time_since_last_play = 0
        self._number_of_real_plays = 0
        #self._squiggles_list = [Squiggles() for i in range(len(self._squiggles_list))]
        #self._squiggles_input = [0 for i in range(16)]
        #for squig in self._squiggles_list:
        #    squig.update_o()

        """
        notes_filled = 1
        for i in range(1,17):
            output = self._squiggles_list[-1].o
            if output[-i] == 1:
                self._time_since_real_plays[-notes_filled] = self._time_between_squiggles_beats*i
                notes_filled+=1
                if notes_filled > len(self._time_since_real_plays):
                    break"""

        return ts.restart(self._time_since_real_plays)

    def _step(self, action):
        if self._episode_ended:
            return self._reset()

        reward = 0

        if action==0:
            if self._time_since_last_play > self._time_between_squiggles_beats*16:
                reward -= 40
            current_closeness_to_real_beat = self._state%self._time_between_squiggles_beats
            reward -= 15*(10**int(-np.absolute(current_closeness_to_real_beat)))-(1/self._time_between_squiggles_beats)

        if action == 1:
            self._number_of_plays += 1
            if self._time_since_last_play < self._time_between_squiggles_beats/3:
                reward -= 10 #Random number, probably needs tweaking

            current_closeness_to_real_beat = self._state%self._time_between_squiggles_beats
            closest_beat = self._state-current_closeness_to_real_beat
            if current_closeness_to_real_beat > self._time_between_squiggles_beats/2:
                closest_beat = self._state + current_closeness_to_real_beat
                current_closeness_to_real_beat -= self._time_between_squiggles_beats
                reward += 0.4

            current_i = int((closest_beat/self._time_between_squiggles_beats)%16)
            self._squiggles_input[current_i] = 1

            reward += 6*(2**int(-np.absolute(current_closeness_to_real_beat)))-1

            self._time_since_last_play = 0

        if action == 1 or action == 0:
            self._state += 1
            self._time_since_real_plays += 1
            self._time_since_last_play += 1
        else:
            raise ValueError('`action` should be 0 or 1.')

        if self._state%(self._time_between_squiggles_beats*16) == 0:
            self._squiggles_list[0].update_h(self._squiggles_input)
            for i in range(1, len(self._squiggles_list)):
                self._squiggles_list[i-1].update_o()
                out_in = self._squiggles_list[i-1].o
                self._squiggles_list[i].update_h(out_in)
            self._squiggles_list[-1].update_o()
            self._squiggles_input = [0 for i in range(16)]

        play = False
        """if self._state%self._time_between_squiggles_beats == 0:
            print("Here I am")
            print((self._state/self._time_between_squiggles_beats)%16)
            print(self._squiggles_list[-1].o)
            print(int((self._state/self._time_between_squiggles_beats)%16))
            print(self._squiggles_list[-1].o[int((self._state/self._time_between_squiggles_beats)%16)])"""
        if self._state%self._time_between_squiggles_beats == 0 and self._squiggles_list[-1].o[int((self._state/self._time_between_squiggles_beats)%16)] == 1:
            #print("I have played", self._state)
            play = True
            self._time_since_real_plays[:-1] = np.roll(self._time_since_real_plays[:-1],1)
            self._time_since_real_plays[0] = 0


        output = 1 if play else 0
        self._number_of_real_plays += output




        self._time_since_real_plays[-1] = self._time_since_last_play

        if self._state >= 6000:
            #50 is a random constant here, should probably be tweaked
            #reward = 10-np.absolute(self._number_of_plays - self._number_of_real_plays)*5
            return ts.termination(self._time_since_real_plays, reward)

        return ts.transition(
            self._time_since_real_plays, reward=reward, discount=0.9)
