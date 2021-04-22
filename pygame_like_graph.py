""" Makes a graph like the pygame_play file (but I will not use pygame) """

from tf_agents.environments import tf_py_environment
from imageio import imread, imsave
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from pygame_play import get_beats
from presicion_test_forall import train_each

##### These change versions we're graphing:
from env.SquigglesEnvironment import SquigglesEnvironment
policy_saved_filename = 'policy_saved'
classifier_name = None # None if you want policy graphed, or:
# "KNeighborsClassifier", "SVC", "MLPClassifier", "RandomForestClassifier"

# Globals
NUM_STEPS = 1000 # this amount of steps will be shown, please do not change
DIST_STEPS = 2   # pixels (shouldn't be 1)

# Box color and dimensions
env_color = (50,180,220)   # rgb
agent_color = (255,150,60)
env_box_dim = (15,13)       # pixels, odd numbers plox
ag_box_dim = (15,13)
bg_color = (255,255,255)
dot_color = (0,0,0)
grid_color = (200,200,200)

# x_margin and y_len, as well as y_env and y_agent can be changed for appearance
x_margin = 10              # pixel margins on each side
x_len = NUM_STEPS*DIST_STEPS + x_margin*2 # pixels
y_len = 200                # pixels
y_env = y_len // 3         # Env line
y_agent = y_len * 2 // 3   # Agent line

def draw_box(image, x,y, color, box_dim):
    n,m = box_dim

    for i in range(-n//2+1, n//2+1):
        for j in range(-m//2+1, m//2+1):
            image[y+i,x+j] = color

    for i in range(-n//2+1-n//3, n//2+1+n//3):
        image[y+i,x] = color

if __name__ == "__main__":
    tempo = 0

    # Get environment and agent playing
    orig_env = SquigglesEnvironment(num_notes=2)
    env = tf_py_environment.TFPyEnvironment(orig_env)

    if classifier_name == None:
        _, the_hits, actions = get_beats(
            env.observation_spec().shape[0],
            NUM_STEPS,
            env,
            policy_saved_filename
        )
    else:
        classifier = train_each(classifier_name, eval(classifier_name))

        the_hits = []
        actions = []

        time_step = env.reset()
        for _ in range(NUM_STEPS):
            a = classifier.predict([time_step.observation[0]])[0]
            actions.append(a)
            time_step = env.step(a)

            # Saving the hits
            play = time_step.observation[0][0] == 0
            the_hits.append(int(play))

    tempo = orig_env._time_between_squiggles_beats

    # Make a long image
    image = np.array([[bg_color for _ in range(x_len)] for _ in range(y_len)])

    # Fill image
    i = 0
    for x in range(x_margin, x_len-x_margin, DIST_STEPS):

        # Make lines
        if (i+1)%tempo == 0:
            for y in range(y_env-30, y_agent+30, round(y_len/20)):
                for j in range(round(y_len/40)):
                    image[y+j,x] = grid_color

        # Make small dots
        if image[y_env,x,0] != env_color[0] or image[y_env,x,1] != env_color[1] or image[y_env,x,2] != env_color[2]:
            image[y_env,x:x+DIST_STEPS] = dot_color
        if image[y_agent,x,0] != agent_color[0] or image[y_agent,x,1] != agent_color[1] or image[y_agent,x,2] != agent_color[2]:
            image[y_agent,x] = dot_color
            if i%10 == 0:
                for y in range(y_agent-2, y_agent):
                    image[y,x] = dot_color
            elif i%5 == 0:
                for y in range(y_agent-1, y_agent):
                    image[y,x] = dot_color

        # Make boxes
        if the_hits[i] == 1:
            draw_box(image, x,y_env, env_color, env_box_dim)
        if actions[i] == 1:
            draw_box(image, x,y_agent, agent_color, ag_box_dim)
        i += 1

    # Show image
    plt.yticks([y_env, y_agent], ["Environment", "Agent" if classifier_name == None else classifier_name])
    plt.xticks(np.arange(x_margin,x_margin+DIST_STEPS*NUM_STEPS, 10*DIST_STEPS), np.arange(0,NUM_STEPS,10))
    plt.imshow(image.astype(int))
    plt.xlabel('Time steps')
    plt.show()

    imsave("long_performance.png", image.astype(int))
