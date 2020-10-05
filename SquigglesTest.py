import numpy as np
from Squiggles import Squiggles
from ComplexRhythmEnvironment import ComplexRhythmEnvironment


test1 = Squiggles()
test2 = Squiggles()


for i in range(10):
    test1.update_o()
    test2.update_h(test1.o)

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
