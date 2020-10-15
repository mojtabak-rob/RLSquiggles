""" File for plotting the environment observation  for one episode, and making
a soundfile of the environment in that episode """

import matplotlib.pyplot as plt
import numpy as np
import wavio # comment out if you can't get wavio

from SquigglesEnvironment import SquigglesEnvironment

## TODO: Expand to also plot and render agent's actions
## TODO: Save agent after cartpole training and load in this file

def get_beats(N, ITER, env):
    state = env.reset()

    beats = [[] for _ in range(N)]
    for _ in range(ITER):
        state = env.step(0)
        for i in range(N):
            beats[i].append(state.observation[i])

    the_hits = np.zeros(ITER)

    for i in range(ITER):
        play = False
        for j in range(N):
            if beats[j][i] == 0:
                play = True
        the_hits[i] = int(play)

    return beats, the_hits

def plot_beats(beats, N, ITER):
    plt.figure()
    times = np.arange(0,ITER,1)
    for i in range(N):
        plt.plot(times, beats[i])

def plot_the_hits(the_hits, ITER):
    plt.figure()
    times = np.arange(0,ITER,1)
    plt.plot(times, the_hits)

def make_soundfile(the_hits, ITER):
    # Source: https://pypi.org/project/wavio/
    # Parameters
    time_step_length = 0.01 #s
    samples = int(1/time_step_length)  # samples per seconds of environment
    rate = int(samples*40)             # samples per second
    T = int(time_step_length*ITER)     # sample duration (seconds)
    f = 240.0                          # sound frequency (Hz)

    # Compute waveform samples
    t = np.linspace(0, T, T*rate, endpoint=False)
    x = np.sin(2*np.pi * f * t)

    # x is now constant sound, I will mute it to make the beat

    i = 0
    double = False # Extending the note for two environment samples to better hear it
    for j in range(ITER):
        if the_hits[j] == 0.0 and not double:
            x[i:i+int(rate / samples)] = 0.0
        elif double:
            double = False
        else:
            double = True
        i += int(rate / samples)

    # Write the samples to a file
    wavio.write("env_sound.wav", x, rate, sampwidth=3)

def main():
    env = SquigglesEnvironment()

    N = env.observation_spec().shape[0]
    ITER = 1000

    beats, the_hits = get_beats(N, ITER, env)

    plot_beats(beats, N, ITER)
    plot_the_hits(the_hits, ITER)
    plt.show()

    make_soundfile(the_hits, ITER) # comment out if you can't get wavio

if __name__ == "__main__":
    main()
