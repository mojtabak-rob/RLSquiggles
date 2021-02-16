""" Fitting and viewing the performance of pre-tuned classifiers
Select which classifier to view by the parameters below the imports.
Ideally would save the trained classifiers """

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import randint

from classifier_util import *

### Parameters for this script
# Change as needed
classifier_name ="RandomForestClassifier"
classifier_class = RandomForestClassifier
num_data_points = 3000 # to train on

if __name__ == "__main__":

    # The hyperparameters are fetched from file
    kwargs = read_hyperparameters(classifier_name)
    print_dict(kwargs)

    # Kwargs initially contains some training parameters too
    balanced = kwargs["balanced_squig"]
    shuffled = kwargs["shuffled_squig"]
    del kwargs["balanced_squig"]
    del kwargs["shuffled_squig"]

    # We continue to fit and predict if the classifier does not produce sound
    continuing = True
    iters = 0
    while continuing:
        if iters != 0:
            print("Classifier stayed silent, iteration number:", iters)

        # Get data
        x_data, y_data = get_balanced_dataset(num_data_points) if balanced else \
                         get_dataset(num_data_points)

        if shuffled:
            x_data, y_data = shuffle_dataset(x_data, y_data)

        # Fit
        classifier = classifier_class(**kwargs)
        classifier.fit(x_data, y_data)

        # Predict
        produced_sound = plot_predict(classifier, classifier_name, 1000)
        continuing = not produced_sound
        iters += 1
