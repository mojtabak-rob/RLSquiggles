#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 12:40:41 2020

@author: mia-katrinkvalsund
"""

# Module imports
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.trajectories import trajectory

# Written by us imports
from SquigglesEnvironment import SquigglesEnvironment
from basic_agent import generic_dqn_agent # a function
from tqdm import tqdm

# Globals
NUMBER_ITERATION = 20000
COLLECTION_STEPS = 1
BATCH_SIZE = 64
EVAL_EPISODES = 5
EVAL_INTERVAL = 1000

#######################################################################

train_env = SquigglesEnvironment()
evaluation_env = SquigglesEnvironment()

train_env = tf_py_environment.TFPyEnvironment(train_env)
evaluation_env = tf_py_environment.TFPyEnvironment(evaluation_env)

agent, _ = generic_dqn_agent(train_env)

#####################################################################

def get_average_return(environment, policy, episodes=10):

    total_return = 0.0

    for _ in tqdm(range(episodes)):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward

        total_return += episode_return
    avg_return = total_return / episodes

    return avg_return.numpy()[0]

#####################################################################

class ExperienceReplay(object):
    def __init__(self, agent, enviroment):
        self._replay_buffer = TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=enviroment.batch_size,
            max_length=50000)

        self._random_policy = RandomTFPolicy(train_env.time_step_spec(),
                                                enviroment.action_spec())

        self._fill_buffer(train_env, self._random_policy, steps=100)

        self.dataset = self._replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=BATCH_SIZE,
            num_steps=2).prefetch(3)

        self.iterator = iter(self.dataset)

    def _fill_buffer(self, enviroment, policy, steps):
        for _ in range(steps):
            self.timestamp_data(enviroment, policy)

    def timestamp_data(self, environment, policy):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        timestamp_trajectory = trajectory.from_transition(time_step, action_step, next_time_step)

        self._replay_buffer.add_batch(timestamp_trajectory)

experience_replay = ExperienceReplay(agent, train_env)

#####################################################################

agent.train_step_counter.assign(0)

avg_return = get_average_return(evaluation_env, agent.policy, EVAL_EPISODES)
returns = [avg_return]

for _ in tqdm(range(NUMBER_ITERATION)):

    for _ in range(COLLECTION_STEPS):
        experience_replay.timestamp_data(train_env, agent.collect_policy)

    experience, info = next(experience_replay.iterator)
    train_loss = agent.train(experience).loss

    if agent.train_step_counter.numpy() % EVAL_INTERVAL == 0:
        avg_return = get_average_return(evaluation_env, agent.policy, EVAL_EPISODES)
        print('Iteration {0} – Average Return = {1}, Loss = {2}.'.format(agent.train_step_counter.numpy(), avg_return, train_loss))
        returns.append(avg_return)

#####################################################################
