import matplotlib.pyplot as plt

from SquigglesEnvironment import SquigglesEnvironment

env = SquigglesEnvironment()
state = env.reset()
print(state)

time = []
beats = [[] for _ in range(len(state.observation))]
print(env._time_between_squiggles_beats)
for _ in range(1000):
    state = env.step(1)
    for i in range(len(state.observation)):
        beats[i].append(state.observation[i])

time = [i for i in range(1000)]
for i in range(len(state.observation)):
    plt.plot(time, beats[i])
plt.show()
