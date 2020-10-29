import numpy as np
import wavio

from Squiggles import Squiggles
from ComplexRhythmEnvironment import ComplexRhythmEnvironment

def make_muted_sine(hits, f, ITER):
    # Source: https://pypi.org/project/wavio/
    # Parameters
    time_step_length = 120/60/16 #s
    samples = int(1/time_step_length)  # samples per seconds of environment
    rate = int(samples*800)             # samples per second
    T = int(time_step_length*ITER)     # sample duration (seconds)

    # Compute waveform samples
    t = np.linspace(0, T, T*rate, endpoint=False)
    x = np.sin(2*np.pi * f * t)

    i = 0
    for j in range(ITER):
        if hits[j] == 0.0:
            x[i:i+int(rate / samples)] = 0.0
        i += int(rate / samples)

    return x, rate

ITER = 120*2

test1 = Squiggles()
test2 = Squiggles()

squig1 = []
squig2 = []
for i in range(ITER):
    squig1.append(test1.o)
    test1.update_o()
    test2.update_h(test1.o)

    squig2.append(test2.o)
    test2.update_o()
    test1.update_h(test2.o)

test1.print_state()
print(test1.ham)
test2.print_state()
print(test2.ham)

"""

envTest = ComplexRhythmEnvironment()
for i in range(500):
    envTest.step(0)

print(envTest.step(1))"""

squig1 = np.array(squig1).ravel()
squig2 = np.array(squig2).ravel()

f_squig1 = 240.0  # sound frequency (Hz)
f_squig2 = 620.0

# Compute waveform samples
x, rate = make_muted_sine(squig1, f_squig1, ITER)

y, _ = make_muted_sine(squig2, f_squig2, ITER)

# A bold assumptions that I can simply add one wave to the other
joint = x + y

# Write the samples to a file
wavio.write("squiggles_test.wav", joint, rate, sampwidth=3)
