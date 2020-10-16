""" File for plotting the environment observation  for one episode, and making
a soundfile of the environment in that episode """

import tensorflow as tf
from tf_agents.environments import tf_py_environment

import matplotlib.pyplot as plt
import numpy as np
import wavio # comment out if you can't get wavio

from SquigglesEnvironment import SquigglesEnvironment

## TODO: Implement an LSTM layer
## TODO: Make a policy that lowers likelyhood after playing 1

def get_beats(N, ITER, env):
    state = env.reset()
    policy = tf.saved_model.load('policy_saved')

    beats = [[] for _ in range(N)]
    actions = []
    the_hits = []

    for _ in range(ITER):
        # Saving action
        a = policy.action(state)
        actions.append(int(a.action[0]))

        # Saving observation
        state = env.step(a)
        for i in range(N):
            beats[i].append(state.observation[0][i]) # Why was it nested?

        # Saving the hits
        play = np.any(state.observation == 0)
        the_hits.append(int(play))

    return beats, the_hits, actions

def plot_beats(beats, N, ITER):
    plt.figure()
    times = np.arange(0,ITER,1)
    for i in range(N):
        plt.plot(times, beats[i], label="Pos "+str(i))
    plt.legend(loc="upper left")

def plot_the_hits(the_hits, actions, ITER):
    plt.figure()
    times = np.arange(0,ITER,1)
    plt.plot(times, the_hits, label="Environment")
    plt.plot(times, actions, label="Agent")
    plt.legend(loc="upper left")
    plt.ylim(0, 1.2)

# Makes a list that is sine where there is a beat, 0 when there is not
# f is frequency
def make_muted_sine(hits, f, ITER):
    # Source: https://pypi.org/project/wavio/
    # Parameters
    time_step_length = 0.01 #s
    samples = int(1/time_step_length)  # samples per seconds of environment
    rate = int(samples*40)             # samples per second
    T = int(time_step_length*ITER)     # sample duration (seconds)

    # Compute waveform samples
    t = np.linspace(0, T, T*rate, endpoint=False)
    x = np.sin(2*np.pi * f * t)

    i = 0
    double = False # Extending the note for two environment samples to better hear it
    for j in range(ITER):
        if hits[j] == 0.0 and not double:
            x[i:i+int(rate / samples)] = 0.0
        elif double:
            double = False
        else:
            double = True
        i += int(rate / samples)

    return x, rate

def make_soundfile(hits, ITER, file_name):
    f = 240.0 # sound frequency (Hz)

    # Compute waveform samples
    x, rate = make_muted_sine(hits, f, ITER)

    # Write the samples to a file
    wavio.write(f"{file_name}.wav", x, rate, sampwidth=3)

def make_joint_soundfile(hits1, hits2, ITER, file_name):
    f_env = 240.0  # sound frequency (Hz)
    f_ag = 620.0

    # Compute waveform samples
    x, rate = make_muted_sine(hits1, f_env, ITER)
    y, _ = make_muted_sine(hits2, f_ag, ITER)

    # A bold assumptions that I can simply add one wave to the other
    joint = x + y

    # Write the samples to a file
    wavio.write(f"{file_name}.wav", joint, rate, sampwidth=3)

def main():
    env = SquigglesEnvironment()
    env = tf_py_environment.TFPyEnvironment(env)

    N = env.observation_spec().shape[0]
    ITER = 1000

    beats, the_hits, actions = get_beats(N, ITER, env)

    plot_beats(beats, N, ITER)
    plot_the_hits(the_hits, actions, ITER)
    plt.show()

    make_soundfile(the_hits, ITER, "env_sound") # comment out if you can't get wavio
    make_soundfile(actions, ITER, "action_sound") # comment out if you can't get wavio
    make_joint_soundfile(the_hits, actions, ITER, "joint_sound") # comment out if you can't get wavio

if __name__ == "__main__":
    main()
