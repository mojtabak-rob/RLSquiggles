# -*- coding: utf-8 -*-
""" Basic code for basic agent. Includes examples of how to use it in loop. 
.train takes an experience from a replay buffer """

import tensorflow as tf
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.environments.tf_py_environment import TFPyEnvironment

from SimpleRhythmEnvironment import SimpleRhythmEnvironment

def generic_dqn_agent(env: TFPyEnvironment) -> (dqn_agent.DqnAgent, q_network.QNetwork):
    """ Function that returns a generic dqn agent
    args:
        env (TFPyEnvironment) : The environment the agent will live in
        
    Returns:
        dqn_agent.DqnAgent: The agent to train
        q_network.QNetwork: The network used in the agent
    """
    q_net = q_network.QNetwork(
      env.observation_spec(),
      env.action_spec(),
      fc_layer_params=(100,))
    
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)
    
    agent = dqn_agent.DqnAgent(
      env.time_step_spec(),
      env.action_spec(),
      q_network=q_net,
      optimizer=optimizer,
      td_errors_loss_fn=common.element_wise_squared_loss,
      train_step_counter=0
    )
    
    agent.initialize()
    
    return agent, q_net

#########################################################################
    
env = SimpleRhythmEnvironment()
env = TFPyEnvironment(env)

####### Example of use ##################################################

agent, net = generic_dqn_agent(env)

s = env.reset()
step_type, reward, discount, observation = s

step = agent.policy.action(s)
state = env.step(step.action)
step2 = agent.policy.action(state)
#state2 = env.step2(step.action)

#agent.train(experience)
