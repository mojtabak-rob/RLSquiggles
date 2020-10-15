import matplotlib.pyplot as plt

from SquigglesEnvironment import SquigglesEnvironment

env = SquigglesEnvironment()
state = env.reset()
print(state)

time = []
beats = []
for _ in range(10000):
    state = env.step(1)
    time.append(state.observation[0])
    beats.append(state.observation[1])

plt.plot(time, beats)
plt.show()
