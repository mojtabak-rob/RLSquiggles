import pygame, sys
from pygame.locals import *
import matplotlib.pyplot as plt
import time
import numpy as np

import tensorflow as tf
from tf_agents.environments import tf_py_environment

from halvors_good_policy.SquigglesEnvironment import SquigglesEnvironment # Change this and policy_saved

# Together, these two control the frequency of played notes.
# Every frame, a new note will be fetched
FPS = 60
STEPS_ACROSS_SCREEN = 6

# Other constants
HEIGHT = 700 # Height and width of screen. All sizes scale with these
WIDTH = 1300
ITER = 3000 # 3 episodes are run

def get_beats(N, ITER, env):
    state = env.reset()
    policy = tf.saved_model.load('halvors_good_policy/policy_saved') # Change this and env

    beats = [[] for _ in range(N)]
    actions = []
    the_hits = []

    for _ in range(ITER):
        # Saving action
        print(state.observation[0][0])
        a = policy.action(state)
        actions.append(int(a.action[0]))

        # Saving observation
        state = env.step(a)
        for i in range(N):
            beats[i].append(state.observation[0][i]) # Why was it nested?

        # Saving the hits
        play = state.observation[0][0] == 0
        the_hits.append(int(play))

    return beats, the_hits, actions

class DrawableRectangle:
    def __init__(self, position_x, position_y, height, width, color):
        self._pos_x = position_x
        self._pos_y = position_y
        self._height = height
        self._width = width
        self._color = color

    def render(self,display):
        pygame.draw.rect(
            display,
            self._color,
            (self._pos_x, self._pos_y, self._width, self._height)
        )

class SoundBarrier(DrawableRectangle):
    def __init__(self, position_x, position_y, height, width, color, slider_list):
        DrawableRectangle.__init__(self, position_x, position_y, height, width, color)
        self._slider_list = slider_list

    def update(self):
        for slider in self._slider_list:
            if slider.has_beat_at(x_pos = self._pos_x+self._width//2):
                slider.play()

class Beat(DrawableRectangle):
    def __init__(self, position_x, position_y, height, width, color):
        DrawableRectangle.__init__(self, position_x, position_y, height, width, color)
        self._played = False

    def update(self, length):
        self._pos_x += length

    def out_of_bounds(self, x_bound):
        if self._pos_x > x_bound:
            return True
        return False

    def is_at(self, x_pos):
        if self._played:
            return False

        if x_pos - self._width//2 < self._pos_x < x_pos + self._width//2:
            self._played = True
            return True

class SoundSlider:
    def __init__(self, sound_list, position_x, position_y, height, width, color, soundfile_name):
        self._my_sounds = sound_list
        self._pos_x = position_x
        self._pos_y = position_y
        self._height = height
        self._width = width
        self._color = color
        self._my_effects = [pygame.mixer.Sound(name) for name in soundfile_name]
        self._effect_counter = 0

        self._counter = 0
        self._current_beats = []

    def play(self):

        self._my_effects[self._effect_counter-1 if self._effect_counter > 0 else -1].stop()
        self._my_effects[self._effect_counter].play()
        self._effect_counter += 1
        if self._effect_counter >= len(self._my_effects):
            self._effect_counter = 0

    def has_beat_at(self, x_pos):
        for beat in self._current_beats:
            if beat.is_at(x_pos):
                return True
        return False

    def render(self, display):
        pygame.draw.line(
            display,
            (255,255,255),
            (self._pos_x, self._pos_y+self._height//2),
            (self._pos_x+self._width, self._pos_y+self._height//2),
            3
        )

        for beat in self._current_beats:
            beat.render(display)

    def update(self):
        for beat in self._current_beats:
            beat.update(length=STEPS_ACROSS_SCREEN)

        for beat in self._current_beats:
            if beat.out_of_bounds(x_bound=self._width+self._pos_x):
                self._current_beats.remove(beat)

        if self._counter >= len(self._my_sounds):
            return None

        if self._my_sounds[self._counter] == 1:
            self._current_beats.append(
                Beat(
                    position_x=self._pos_x,
                    position_y=self._pos_y+self._height//2-15,
                    height=30,
                    width=15,
                    color=self._color
                )
            )

        self._counter += 1

def main():
    env = SquigglesEnvironment()
    env = tf_py_environment.TFPyEnvironment(env)

    N = env.observation_spec().shape[0]

    _, the_hits, actions = get_beats(N, ITER, env)

    fpsClock = pygame.time.Clock()
    pygame.init()

    DISPLAY = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Squigs")

    """ Here's different sounds to use
    ,
    "sound_effects/19827__cabled-mess__glockenspiel/348882__cabled-mess__glockenspiel-18-g3-04.wav",
    "sound_effects/19827__cabled-mess__glockenspiel/348889__cabled-mess__glockenspiel-23-a3-05.wav",
    "sound_effects/19827__cabled-mess__glockenspiel/348895__cabled-mess__glockenspiel-24-bb3-01.wav",
    "sound_effects/19827__cabled-mess__glockenspiel/348904__cabled-mess__glockenspiel-29-b3-02.wav",
    "sound_effects/19827__cabled-mess__glockenspiel/348914__cabled-mess__glockenspiel-39-d4-04.wav",
    "sound_effects/19827__cabled-mess__glockenspiel/348918__cabled-mess__glockenspiel-40-e4-01.wav",
    "sound_effects/19827__cabled-mess__glockenspiel/348921__cabled-mess__glockenspiel-43-f4-01.wav"
    "sound_effects/19827__cabled-mess__glockenspiel/348870__cabled-mess__glockenspiel-04-d3-04.wav",
    "sound_effects/19827__cabled-mess__glockenspiel/348871__cabled-mess__glockenspiel-06-e3-01.wav",
    "sound_effects/19827__cabled-mess__glockenspiel/348878__cabled-mess__glockenspiel-11-f3-02.wav",
    "sound_effects/19827__cabled-mess__glockenspiel/348908__cabled-mess__glockenspiel-33-c4-02.wav"
    """

    """
    "sound_effects/9008__jamieblam__metallophone/146077__jamieblam__1d-hard.wav"
    "sound_effects/9008__jamieblam__metallophone/146079__jamieblam__1c-hard.wav"
    """

    """
    "sound_effects/21030__samulis__vsco-2-ce-percussion-marimba/373577__samulis__marimba-b3-marimba-hit-outrigger-b2-loud-01.wav"
    "sound_effects/21030__samulis__vsco-2-ce-percussion-marimba/373582__samulis__marimba-e-2-marimba-hit-outrigger-f1-loud-01.wav"
    """

    """
    ,
    "sound_effects/9008__jamieblam__metallophone/146096__jamieblam__2e-hard.wav",
    "sound_effects/9008__jamieblam__metallophone/146100__jamieblam__2f-hard.wav"
    """

    env_slider = SoundSlider(
        sound_list = the_hits,
        position_x = 0,
        position_y = HEIGHT//3,
        height = HEIGHT//7,
        width = WIDTH,
        color = (100,100,255),
        soundfile_name = [
            "sound_effects/9008__jamieblam__metallophone/146079__jamieblam__1c-hard.wav",
            "sound_effects/9008__jamieblam__metallophone/146077__jamieblam__1d-hard.wav",
            "sound_effects/9008__jamieblam__metallophone/146084__jamieblam__1e-hard.wav",
            "sound_effects/9008__jamieblam__metallophone/146082__jamieblam__1f-hard.wav",
            "sound_effects/9008__jamieblam__metallophone/146087__jamieblam__1g-hard.wav"
        ]
    )
    agent_slider = SoundSlider(
        sound_list = actions,
        position_x = 0,
        position_y = HEIGHT*2//3,
        height = HEIGHT//7,
        width = WIDTH,
        color = (255,150,30),
        soundfile_name = [
            "sound_effects/9008__jamieblam__metallophone/146093__jamieblam__2b-hard.wav",
            "sound_effects/9008__jamieblam__metallophone/146091__jamieblam__2c-hard.wav",
            "sound_effects/9008__jamieblam__metallophone/146097__jamieblam__2d-hard.wav"
        ]
    )

    barrier = SoundBarrier(
        position_x = WIDTH*2//3,
        position_y = HEIGHT//4,
        height = HEIGHT*5//8,
        width = WIDTH//56,
        color = (255,100,100),
        slider_list = [env_slider, agent_slider]
    )

    while True:
        DISPLAY.fill((0,0,0))
        pygame.event.pump()

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        env_slider.update()
        agent_slider.update()
        barrier.update()

        env_slider.render(DISPLAY)
        agent_slider.render(DISPLAY)
        barrier.render(DISPLAY)

        pygame.display.update()
        fpsClock.tick(FPS)

if __name__ == "__main__":
    main()
