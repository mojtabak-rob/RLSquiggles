import matplotlib.pyplot as plt
import numpy as np

from SquigglesEnvironment import SquigglesEnvironment

env = SquigglesEnvironment()
state = env.reset()
print(state)

N = env.observation_spec().shape[0]
ITER = 1000

beats = [[] for _ in range(N)]
for _ in range(ITER):
    state = env.step(0)
    for i in range(N):
        beats[i].append(state.observation[i])

times = np.arange(0,ITER,1)
for i in range(N):
    plt.plot(times, beats[i])

plt.figure()
the_hits = np.zeros(ITER)

for i in range(ITER):
    play = False
    for j in range(N):
        if beats[j][i] == 0:
            play = True
    the_hits[i] = int(play)

plt.plot(times, the_hits)
plt.show()