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
from tf_agents.policies.policy_saver import PolicySaver

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Written by us imports
from env.SquigglesEnvironment import SquigglesEnvironment
#from versions.mirror_no_silence_punish.SquigglesEnvironment import SquigglesEnvironment
from experience_replay import ExperienceReplay
from basic_agent import generic_dqn_agent # a function

# Globals
NUMBER_ITERATION = 30000
COLLECTION_STEPS = 1
BATCH_SIZE = 64
EVAL_EPISODES = 3
EVAL_INTERVAL = 1000

def get_average_return(environment, policy, episodes=10):

    np.random.seed(42)

    total_return = 0.0
    average_played = 0

    for _ in range(episodes):
        time_step = environment.reset()
        episode_return = 0.0

        times_played = 0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            if action_step.action[0] == 1:
                times_played += 1

            time_step = environment.step(action_step.action)
            episode_return += time_step.reward

        average_played += times_played / 1000
        total_return += episode_return
    avg_return = total_return / episodes

    np.random.seed(None)

    print("Average percent played:", average_played/episodes)

    return avg_return.numpy()[0]

def init():
    train_env = SquigglesEnvironment(num_notes=2)
    evaluation_env = SquigglesEnvironment(num_notes=2)

    train_env = tf_py_environment.TFPyEnvironment(train_env)
    evaluation_env = tf_py_environment.TFPyEnvironment(evaluation_env)

    agent, _ = generic_dqn_agent(train_env)

    experience_replay = ExperienceReplay(agent, train_env, BATCH_SIZE)

    return agent, train_env, evaluation_env, experience_replay

def training_loop(agent, train_env, evaluation_env, experience_replay):
    agent.train_step_counter.assign(0)

    avg_return = get_average_return(evaluation_env, agent.policy, EVAL_EPISODES)
    returns = [avg_return]

    for iteration in tqdm(range(NUMBER_ITERATION)):

        for _ in range(COLLECTION_STEPS):
            experience_replay.timestamp_data(train_env, agent.collect_policy)

        experience, info = next(experience_replay.iterator)
        train_loss = agent.train(experience).loss

        if agent.train_step_counter.numpy() % EVAL_INTERVAL == 0:
            avg_return = get_average_return(evaluation_env, agent.policy, EVAL_EPISODES)
            print('Iteration {0} – Average Return = {1}, Loss = {2}.'.format(agent.train_step_counter.numpy(), avg_return, train_loss))
            returns.append(avg_return)

            #show_current(1000, evaluation_env, agent.policy)

    return returns

def show_current(ITER, env, policy):
    N = env.observation_spec().shape[0]
    state = env.reset()

    the_hits = np.zeros(ITER)
    agent_hits = []
    rewards = []
    for j in range(ITER):
        a = policy.action(state)
        agent_hits.append(a.action)

        state = env.step(a)
        rewards.append(state.reward)

        play = False
        if np.any(state.observation[0][0] == 0):
            play = True
        the_hits[j] = int(play)

    plt.figure()
    plt.plot(the_hits)
    plt.plot(agent_hits)
    plt.title("Action and space")

    plt.figure()
    plt.plot(rewards)
    plt.title("Rewards")
    plt.show()

def main():
    runs = 5
    agent, train_env, evaluation_env, experience_replay = init()

    returns = training_loop(
        agent,
        train_env,
        evaluation_env,
        experience_replay
    )

    # save policy
    PolicySaver(agent.policy).save('policy_saved')

    plt.plot([i for i in range(0,NUMBER_ITERATION+1, EVAL_INTERVAL)],returns)
    plt.title("Rewards overall")
    plt.show()

if __name__ == "__main__":
    main()
