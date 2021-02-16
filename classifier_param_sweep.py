""" Sweep over a select number of params, writing the best performing combination to file.

The following explains the tuning:

Training set is balanced by under-sampling the 0-label class.
This is done because unbalanced datasets fail to produce any non-zero actions in classifiers.

Sometimes, I will use a heuristic, other times I use random search.
This is because I lack the knowledge to choose a good grid, but a random search
can often compete with a grid.
https://jmlr.csail.mit.edu/papers/volume13/bergstra12a/bergstra12a.pdf

SVM heuristic from:
https://www.uio.no/studier/emner/matnat/ifi/IN5520/h20/undervisningsmateriale/lecturenotes/in5520_svm_2020.pdf

RandomForestClassifier heuristic from observation and:
https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import randint
import copy

from classifier_util import *

### Adjust here
# Comment out the classifiers you don't want to tune
to_sweep = [
    #"KNeighborsClassifier",
    #"MLPClassifier",
    #"SVC",
    "RandomForestClassifier"
]

# Meticulously chosen parameters to try
# If you add more, bear in mind the number of combinations it must now try
params = {
    "RandomForestClassifier" : { # 144 combinations
        'max_features': ['auto'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        "n_estimators":[np.random.randint(10,200) for _ in range(4)],
        "max_depth":[None]+[np.random.randint(20,100) for _ in range(3)], # In my experience, high max_depth is better
        'bootstrap': [False], # With a very small test, I decided False was best, but it varied a lot
        "balanced_squig":[True],
        "shuffled_squig":[True]
    },
    "SVC" : { # 110 combinations
        "C":[2**i for i in range(-5,16,2)],
        "gamma":[2**i for i in range(-15,4,2)], # Using rbf kernel
        "balanced_squig":[True],
        "shuffled_squig":[False]
    },
    "MLPClassifier" : { # 20 combinations
        "hidden_layer_sizes":[(np.random.randint(10,200),) for _ in range(5)],
        "activation":["identity", "logistic", "tanh", "relu"],
        "balanced_squig":[True],
        "shuffled_squig":[True]
    },
    "KNeighborsClassifier" : { # 14 combinations
        "n_neighbors":[1, 3, 5, 7, 13, 17, 23], # Heuristic from observation
        "weights":["uniform", "distance"],
        "n_jobs":[1],               # For running parallell jobs
        "balanced_squig":[True],
        "shuffled_squig":[False]
    }
}

def recursive_test(to_tune, i, kwargs, c_n, c_c, t):
    if i >= len(to_tune):
        print("Testing:",kwargs)
        t.append(
            [score(c_c, kwargs), copy.copy(kwargs)]
        )
        print("Score:", t[-1][0])
    else:
        for p in params[classifier_name][to_tune[i]]:
            kwargs[to_tune[i]] = p
            recursive_test(to_tune, i+1, kwargs, c_n, c_c, t)

for classifier_name in to_sweep:
    print("Sweep for:", classifier_name)
    classifier_class = eval(classifier_name)

    test_scores = []
    to_tune = list(params[classifier_name].keys())

    recursive_test(to_tune, 0, {}, classifier_name, classifier_class, test_scores)

    # Write to file
    with open("classifier_hyper_params/{0}.txt".format(classifier_name), "w") as file:
        best = [0.0, {}]
        for elem in test_scores:
            if elem[0] > best[0]:
                best = elem

        balanced = best[1]["balanced_squig"]
        shuffled = best[1]["shuffled_squig"]
        del best[1]["balanced_squig"]
        del best[1]["shuffled_squig"]

        file.write("# Hyperparameters for "+classifier_name+"\n")
        file.write("# Minimum precision: " + str(best[0])+"\n")
        for p in best[1].keys():
            file.write(str(p)+" = "+str(best[1][p])+"\n")

        file.write("\n# For the training\n")
        file.write("balanced_squig = "+str(balanced)+"\n")
        file.write("shuffled_squig = "+str(shuffled)+"\n")

    print(classifier_name, "done")
