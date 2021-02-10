""" TODO. Sweep over a select number of params, writing the best performing ones to file """

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import randint

from classifier_util import *

# On all four classifiers
    # Sweep over selected parameters in a range
        # Calculate score for one configuration: Average?
        # Save score

    # Choose best score and save params to file
