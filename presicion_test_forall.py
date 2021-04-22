from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import randint

import tensorflow as tf
from tf_agents.environments import tf_py_environment

from classifier_util import *
from env.SquigglesEnvironment import SquigglesEnvironment

num_data_points = 40000

def train_each(classifier_name, classifier_class):
    # The hyperparameters are fetched from file
    kwargs = read_hyperparameters(classifier_name)
    print_dict(kwargs)

    # Kwargs initially contains some training parameters too
    balanced = kwargs["balanced_squig"]
    shuffled = kwargs["shuffled_squig"]
    del kwargs["balanced_squig"]
    del kwargs["shuffled_squig"]

    # Get data
    np.random.seed(222)
    x_data, y_data = get_balanced_dataset(num_data_points) if balanced else \
                     get_dataset(num_data_points)
    np.random.seed(None)

    if shuffled:
        x_data, y_data = shuffle_dataset(x_data, y_data)

    # Fit
    classifier = classifier_class(**kwargs)
    classifier.fit(x_data, y_data)

    return classifier

if __name__ == "__main__":
    rfc = train_each("RandomForestClassifier", RandomForestClassifier)
    mlp = train_each("MLPClassifier", MLPClassifier)
    knn = train_each("KNeighborsClassifier", KNeighborsClassifier)
    svc = train_each("SVC", SVC)

    # Current best model
    policy = tf.saved_model.load('versions/version_2D_mirror/policy_saved')

    # Predict
    n = 10
    score = np.zeros((n,5))
    for trial_nr in tqdm(range(n)):
        ITER = 1000
        env = SquigglesEnvironment(num_notes=2)
        env = tf_py_environment.TFPyEnvironment(env)
        time_step = env.reset()

        confusion = [[[0,0],[0,0]] for _ in range(5)]
        for _ in range(ITER):
            obs = time_step.observation[0]
            a_right = label(obs)

            policy_reaction = policy.action(time_step)
            a = policy_reaction.action[0]
            confusion[0][a_right][a] += 1

            a = rfc.predict([time_step.observation[0]])[0]
            confusion[1][a_right][a] += 1

            a = mlp.predict([time_step.observation[0]])[0]
            confusion[2][a_right][a] += 1

            a = knn.predict([time_step.observation[0]])[0]
            confusion[3][a_right][a] += 1

            a = svc.predict([time_step.observation[0]])[0]
            confusion[4][a_right][a] += 1

            time_step = env.step(a_right)

        for i in range(5):
            all_0 = confusion[i][0][0] + confusion[i][0][1]
            all_1 = confusion[i][1][0] + confusion[i][1][1]

            presicion_0 = confusion[i][0][0] / all_0
            presicion_1 = confusion[i][1][1] / all_1

            score[trial_nr,i] = min(presicion_0, presicion_1)

    # Printing so I can easily copy into Latex
    print("Policy RFC MLP KNN SVC")

    n,m = len(score), len(score[0])
    for i in range(n):
        print(i+1, end=" ")
        for j in range(m):
            print("&", round(score[i][j], 4), end=" ")

        print("\\\\")
