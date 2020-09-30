import numpy as np

class Squiggles():
    """Squiggles simulation based on Mike's environment"""
    def __init__(self):
        self.hist = [np.random.rand() for i in range(16)]
        self.o = [0 for i in range(16)]
        self.o_prev = [0 for i in range(16)]
        self.d = 0.9 #Discount
        self.i = 0   #Inverse
        self.k = 0   #Exponent
        self.ham = -1

    def print_state(self):
        print(self.hist)
        print(self.o)

    """def f(self, x, exponent, is_inverse):
        is_inverse = -1 if is_inverse else 0


        nonliniarity = np.abs(2*x-1)
        nonliniarity = nonliniarity**exponent
        nonliniarity *= is_inverse
        if x < 0.5:
            nonliniarity *= -1
        nonliniarity = nonliniarity/2

        return nonliniarity

    def determancy(self):
        result = 0
        minimax = 2
        maximin = -1

        for i in range(16):
            if self.hist[i] >= 0.5:
                if self.hist[i] < minimax:
                    minimax = self.hist[i]
            else:
                if self.hist[i] > maximin:
                    maximin = self.hist[i]

        if minimax == 2 and maximin == -1:
            result = 0
        else:
            result = np.abs(self.f(minimax, self.k, self.i) - self.f(maximin, self.k, self.i))

        return result"""

    def update_h(self, listening_to):
        for i in range(16):
            self.hist[i] = self.hist[i]*self.d + listening_to[i]*(1-self.d)

    def update_o(self):
        self.ham = 0
        for i in range(16):
            self.o_prev[i] = self.o[i]
            test = np.random.rand()
            self.o[i] = 1 if self.hist[i] > test else 0 #self.f(self.hist[i], self.k, self.i) > test else 0
            self.ham += 1 if self.o_prev[i] == self.o[i] else 0
