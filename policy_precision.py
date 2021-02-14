""" This script is not here to stay, but:
Script to calculate the minimum precision of the saved policy """

import tensorflow as tf
from tf_agents.environments import tf_py_environment

from versions.mirror_no_silence_punish.SquigglesEnvironment import SquigglesEnvironment
from classifier_util import label

policy = tf.saved_model.load('versions/mirror_no_silence_punish/policy_saved')

# Predict
n = 3
presicion_0 = 0
presicion_1 = 0
for _ in range(n):
    ITER = 1000
    env = SquigglesEnvironment()
    env = tf_py_environment.TFPyEnvironment(env)
    time_step = env.reset()

    confusion = [[0,0],[0,0]]
    for _ in range(ITER):
        policy_reaction = policy.action(time_step)
        a = policy_reaction.action[0]

        obs = time_step.observation[0]
        a_right = label(obs)
        time_step = env.step(a_right)

        confusion[a_right][a] += 1

    all_0 = confusion[0][0] + confusion[0][1]
    all_1 = confusion[1][0] + confusion[1][1]

    presicion_0 += confusion[0][0] / all_0
    presicion_1 += confusion[1][1] / all_1
presicion_0 = presicion_0 / n
presicion_1 = presicion_1 / n

print("Score:", min(presicion_0, presicion_1))
