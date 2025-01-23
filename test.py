import sys
from main import ENVI 
import gymnasium, torch
from torch.distributions import Categorical
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from model import ActorNetwork

app = QApplication.instance()
if app is None:
    app = QApplication()

actor = ActorNetwork()
actor.load_state_dict(torch.load("data/actor.pth"))

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
    def __init__(self,render:bool):
        self.render = render
        self.terminated = False
        self.observation = None
        self.action = None
        self.env = env
        self.timer = QTimer()

    def stepComputing(self):
        self.action = action_generator()
        self.observation, self.reward, self.terminated, self.truncated, self.info = self.env.step(self.action)

    def guiRendering(self):
            if not self.terminated:
                self.stepComputing()
                self.env.render()
            else:
                self.timer.stop()
                sys.exit()

    def run(self):
        if self.render:
            self.env.reset()
            self.timer.timeout.connect(self.guiRendering)
            self.timer.start(100)
            app.exec()
        else:
            while not self.terminated:
                self.stepComputing()
                print(f"Action : {self.action} | reward : {self.reward}")

     
test = Test(render=True) # Setting render to false will lead to faster computing obviously.
test.run()


