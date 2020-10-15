#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 12:40:41 2020
"""

# Module imports
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.trajectories import trajectory

import matplotlib.pyplot as plt
from tqdm import tqdm

# Written by us imports
from SquigglesEnvironment import SquigglesEnvironment
from experience_replay import ExperienceReplay
from basic_agent import generic_dqn_agent # a function

# Globals
NUMBER_ITERATION = 2000
COLLECTION_STEPS = 1
BATCH_SIZE = 64
EVAL_EPISODES = 5
EVAL_INTERVAL = 100

def get_average_return(environment, policy, episodes=10):

    total_return = 0.0

    for _ in range(episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward

        total_return += episode_return
    avg_return = total_return / episodes

    return avg_return.numpy()[0]

def init():
    train_env = SquigglesEnvironment()
    evaluation_env = SquigglesEnvironment()

    train_env = tf_py_environment.TFPyEnvironment(train_env)
    evaluation_env = tf_py_environment.TFPyEnvironment(evaluation_env)

    agent, _ = generic_dqn_agent(train_env)

    experience_replay = ExperienceReplay(agent, train_env, BATCH_SIZE)

    return agent, train_env, evaluation_env, experience_replay

def training_loop(agent, train_env, evaluation_env, experience_replay):
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
    return returns

def main():
    agent, train_env, evaluation_env, experience_replay = init()

    returns = training_loop(
        agent,
        train_env,
        evaluation_env,
        experience_replay
    )

    plt.plot(returns)
    plt.show()

if __name__ == "__main__":
    main()
