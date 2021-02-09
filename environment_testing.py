import tensorflow as tf
from tf_agents.environments import tf_py_environment

import matplotlib.pyplot as plt
import numpy as np
import wavio

from versions.mirror_no_silence_punish.SquigglesEnvironment import SquigglesEnvironment

# A way to label mirror_no_silence_punish's environment observations
def label(observation):
    sixteenth = observation[-1]

    if observation[0]%sixteenth == 0:
        return 1
    return 0

env = SquigglesEnvironment()
env = tf_py_environment.TFPyEnvironment(env)

N = env.observation_spec().shape[0]-1 # Last observation is action, does not change labelling
ITER = 10000

obs = [[],[]] # One list for each action

# Collecting observations
time_step = env.reset()

for _ in range(ITER):
    obs_here = []
    for i in range(N):
        obs_here.append(
            int(time_step.observation[0][i])
        )
    a = label(np.array(obs_here))
    obs[a].append(obs_here)

    # The labels are fed to the env, in an attempt to not affect the squiggles in the env
    time_step = env.step(a)

# Transpose from ITERx3 to 3xITER
zero = np.transpose(obs[0])
ones = np.transpose(obs[1])

# 2D plot
fig = plt.figure()
plt.scatter(zero[0], zero[1], marker="o", label="Right answer is 0")
plt.scatter(ones[0], ones[1], marker="x", label="Right answer is 1")
plt.legend()
plt.xlabel('First counter')
plt.ylabel('Second counter')
plt.show()

# 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(zero[0], zero[1], zero[2], marker="o", label="Right answer is 0")
ax.scatter(ones[0], ones[1], ones[2], marker="x", label="Right answer is 1")

ax.set_xlabel('First counter')
ax.set_ylabel('Second counter')
ax.set_zlabel('Difference')
plt.legend()

plt.show()
