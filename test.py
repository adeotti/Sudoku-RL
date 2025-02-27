import sys
from main import ENVI 
import gymnasium, torch
from torch.distributions import Categorical
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
import torch
import torch.nn.functional as F
import torch.nn as nn

app = QApplication.instance()
if app is None:
    app = QApplication()

class ActorNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.size = 81
    self.action_dist = 27
    self.action_spec = (3,9)

    self.input_layer = nn.LazyLinear(81)
    self.flat = nn.Flatten()
    self.dense_one = nn.Linear(self.size,self.size)
    self.dense_two = nn.Linear(self.size,self.size)
    self.output = nn.Linear(self.size,self.action_dist)

  def forward(self,x):
    x = self.flat(x)
    x = F.relu(self.input_layer(x))
    #x = torch.flatten(x)
    x = F.relu(self.dense_one(x))
    x = F.relu(self.dense_two(x))
    x = F.relu(self.output(x))
    x = torch.unflatten(x,-1,(self.action_spec))
    return F.softmax(x,-1)
  
actor = ActorNetwork()
actor.load_state_dict(torch.load("trainingData/100k_v2/actor_100k.pth"))

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
                print(f"{self.action}|{self.reward}")
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


