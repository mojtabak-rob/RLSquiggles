""" Makes a graph like the pygame_play file (but I will not use pygame) """

from tf_agents.environments import tf_py_environment
from imageio import imread, imsave
import numpy as np
import matplotlib.pyplot as plt

from pygame_play import get_beats

##### These change versions we're graphing:
from versions.mirror_no_silence_punish.SquigglesEnvironment import SquigglesEnvironment
policy_saved_filename = 'versions/mirror_no_silence_punish/policy_saved'

# Globals
NUM_STEPS = 1000
DIST_STEPS = 2 # pixels

# Box color and dimensions
env_color = (100,100,255) # rgb
agent_color = (255,150,30)
env_box_dim = (15,13) # pixels, odd numbers plox
ag_box_dim = (15,13)

# x_margin and y_len, as well as y_env and y_agent can be changed for appearance
x_margin = 10 # pixel margins on each side
x_len = NUM_STEPS*DIST_STEPS + x_margin*2 # pixels
y_len = 200 # pixels
y_env = y_len // 3         # Env line
y_agent = y_len * 2 // 3   # Agent line

def draw_box(image, x,y, color, box_dim):
    n,m = box_dim

    for i in range(-n//2, n//2):
        for j in range(-m//2, m//2):
            image[y+i,x+j] = color

if __name__ == "__main__":
    # Get environment and agent playing
    env = tf_py_environment.TFPyEnvironment(SquigglesEnvironment())
    _, the_hits, actions = get_beats(
        env.observation_spec().shape[0],
        NUM_STEPS,
        env,
        policy_saved_filename
    )

    # Make a long image
    image = np.zeros((y_len, x_len, 3))

    # Fill image
    i = 0
    for x in range(x_margin, x_len-x_margin, DIST_STEPS):

        # Make small white dots
        if np.all(image[y_env,x] == 0):
            image[y_env,x] = (255,255,255)
        if np.all(image[y_agent,x] == 0):
            image[y_agent,x] = (255,255,255)

        # Make boxes
        if the_hits[i] == 1:
            draw_box(image, x,y_env, env_color, env_box_dim)
        if actions[i] == 1:
            draw_box(image, x,y_agent, agent_color, ag_box_dim)
        i += 1

    # Show image
    plt.imshow(image.astype(int))
    plt.show()
