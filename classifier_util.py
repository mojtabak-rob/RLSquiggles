""" A file for various utility functions connected to the training of the classifiers """
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import randint

from visualize import make_joint_soundfile
from env.SquigglesEnvironment import SquigglesEnvironment
#from versions.mirror_no_silence_punish.SquigglesEnvironment import SquigglesEnvironment

## A way to label environment observations
# OBS: 2 versions:
# 1    means mirroring, predictive
# 2    means sixteenth notes, not-predictive
#
def label(observation):
    sixteenth = observation[len(observation)//2:-1]

    # 2
    # if 0 in sixteenth:
    #    return 1
    # elif observation[0]%sixteenth == 0:
    #     for s in sixteenth:
    #         if observation[0]%s == 0:
    #             return 1
    #     return 0

    # 1
    if observation[0]+1 in sixteenth:
        return 1
    return 0

def score(classifier_class, kwargs):
    # Kwargs initially contains some training parameters too
    balanced = kwargs["balanced_squig"]
    shuffled = kwargs["shuffled_squig"]
    del kwargs["balanced_squig"]
    del kwargs["shuffled_squig"]

    np.random.seed(97)

    # Get data
    x_data, y_data = get_balanced_dataset(10000,verbose=False) if balanced else \
                     get_dataset(10000,verbose=False)
    if shuffled:
        x_data, y_data = shuffle_dataset(x_data, y_data,verbose=False)

    # Fit
    classifier = classifier_class(**kwargs)
    classifier.fit(x_data, y_data)

    # Predict
    n = 5
    presicion_0 = 0
    presicion_1 = 0
    for _ in range(n):
        confusion = get_confusion(classifier)

        all_0 = confusion[0][0] + confusion[0][1]
        all_1 = confusion[1][0] + confusion[1][1]

        presicion_0 += confusion[0][0] / all_0
        presicion_1 += confusion[1][1] / all_1
    presicion_0 = presicion_0 / n
    presicion_1 = presicion_1 / n

    kwargs["balanced_squig"] = balanced
    kwargs["shuffled_squig"] = shuffled

    np.random.seed(None)

    return min(presicion_0, presicion_1)

# A confusion matrix on the form [gold standard, classifier prediction]
def get_confusion(classifier):
    ITER = 1000
    env = SquigglesEnvironment()
    time_step = env.reset()

    confusion = [[0,0],[0,0]]
    for _ in range(ITER):
        obs = time_step.observation
        a = classifier.predict([obs])
        a_right = label(obs)
        time_step = env.step(a_right)

        confusion[a_right][a[0]] += 1

    return confusion

def read_hyperparameters(classifier_name):
    kwargs = {}
    with open("classifier_hyper_params_mirror2D/"+classifier_name+".txt", "r") as file:
        for line in file:

            # Allowing comments
            if line[0] == "#" or len(line.strip()) == 0:
                continue
            words = [item.strip() for item in line.split("=")]

            if len(words) != 2:
                print("Warning: Wrong statement in parameter file: \"{0}\"".format(line))
                print("We allow only: variable_name = value")
                continue
            else:
                key = words[0]
                value = words[1]

            # We can't know if a param is str / int / float / bool / tuple
            try:
                # if int, float, bool, tuple, or list
                kwargs[key] = eval(value)
            except:
                # if str
                kwargs[key] = value

    return kwargs

# Run the environment to get data and label it
def get_dataset(num_data_points, verbose=True):
    if verbose:
        print("Getting data...")
    env = SquigglesEnvironment()

    x_data = []
    y_data = []

    train_step = env.reset()
    ones, zeros = 0, 0

    r = tqdm(range(num_data_points)) if verbose else range(num_data_points)
    for _ in r:
        obs = train_step.observation
        a = label(obs)
        if a == 0:
            zeros += 1
        else:
            ones += 1

        x_data.append(obs)
        y_data.append(a)

        train_step = env.step(a)

    if verbose:
        print("Collected", ones*100/num_data_points, "% 1-samples and ", zeros*100/num_data_points, "% 0-samples")

    return x_data, y_data

# Shuffle the observations and labels
def shuffle_dataset(x_data, y_data, verbose=True):
    if verbose:
        print("Shuffeling data...")

    copy_x = []
    copy_y = []

    num_data_points = len(x_data)

    for i in range(num_data_points):
        index = randint(0,len(x_data)-1)
        copy_x.append(x_data[index])
        copy_y.append(y_data[index])

        x_data.pop(index)
        y_data.pop(index)

    return copy_x, copy_y

def get_balanced_dataset(num_data_points, verbose=True):
    if verbose:
        print("Getting balanced data...")

    x_ones, x_zeros = [], []
    y_ones, y_zeros = [], []

    # We make sure that we collect as many 0-samples as 1-samples from each sample
    allowed_to_add = 0

    while len(x_ones) + len(x_zeros) < num_data_points:
        x_data, y_data = get_dataset(num_data_points, verbose=False)
        x_data, y_data = shuffle_dataset(x_data, y_data, verbose=False)

        # Noisy generator object or not?
        r = tqdm(range(num_data_points)) if verbose else range(num_data_points)

        for i in r:
            if len(x_ones) < num_data_points/2 and y_data[i] == 1:
                x_ones.append(x_data[i])
                y_ones.append(1)
                allowed_to_add += 1

            if len(x_zeros) < num_data_points/2 and y_data[i] == 0 and allowed_to_add > 0:
                x_zeros.append(x_data[i])
                y_zeros.append(0)
                allowed_to_add -= 1

    x_data = x_zeros + x_ones
    y_data = y_zeros + y_ones

    if verbose:
        print("Collected", len(x_ones)*100/num_data_points, "% 1-samples and ", len(x_zeros)*100/num_data_points, "% 0-samples")

    return x_data, y_data

def predict_on_env(classifier, ITER):
    env = SquigglesEnvironment()
    time_step = env.reset()

    the_hits = []
    actions = []
    for _ in range(ITER):
        obs = time_step.observation
        a = classifier.predict([obs])
        time_step = env.step(a)

        actions.append(a)
        the_hits.append(
            obs[0] == 0
        )
    return the_hits, actions

def plot_predict(classifier, classifier_name, ITER):
    the_hits, actions = predict_on_env(classifier, ITER)

    # Returns true and plots if an action was performed
    # Returns false if not, no plots
    found = False
    for a in actions:
        if a == 1:
            found = True
    if not found:
        return False

    time = np.arange(ITER)

    make_joint_soundfile(the_hits, actions, ITER, "output/"+classifier_name+"_joint")

    plt.figure()
    plt.plot(time, the_hits)
    plt.plot(time, actions)
    plt.title(classifier_name)
    plt.show()

    return True

def print_dict(d):
    for key in d.keys():
        print(key, d[key], type(d[key]))
