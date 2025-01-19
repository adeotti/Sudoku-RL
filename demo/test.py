# By default, the testing will be perform on the smaller (9) environment with the smaller model
import gymnasium, torch
from torch.distributions import Categorical
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

"""from board_9.model_9 import pnetwork
from board_9.gui import ENVI"""

app = QApplication.instance()
if app is None:
    app = QApplication()
 
actor.load_state_dict(torch.load("models/actor.pth"))

env = gymnasium.make("sudoku")

def format_observation(env = env):
    observation = env.reset()[0]
    observation = torch.tensor(observation).float().unsqueeze(0)
    return observation

def action_generator(actor = actor):
    obs = format_observation()
    action = actor(obs)
    action = Categorical(action).sample()[0].tolist()
    return action

class Test:
    def __init__(self):
        self.terminated = False
        self.observation = None
        self.action = None
        self.env = env

        self.timer = QTimer()

    def main(self):
        action = action_generator()
        print(f"action | {action}")
        self.env.step(action)
        _,_,terminated,_,_ = self.env.step(action)
        self.terminated = terminated
        self.env.render()
        if self.terminated:
            self.timer.stop()
            
    def run(self):
        self.timer.timeout.connect(self.main)
        self.timer.start(100)
        app.exec()

test = Test()
test.run()


