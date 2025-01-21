import sys
sys.path.append("..")
from main import ENVI 
import gymnasium, torch
from torch.distributions import Categorical
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
#rom model import ActorNetwork

import torch.nn.functional as F
import torch.nn as nn


app = QApplication.instance()
if app is None:
    app = QApplication()

class ActorNetwork(nn.Module):
  # In order to incorporate a mcts search into the system maybe the actor network should output
  # many probability distribution instead of just one 

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
    x = F.relu(self.dense_one(x))
    x = F.relu(self.dense_two(x))
    x = F.relu(self.output(x))
    x = torch.unflatten(x,-1,(self.action_spec))
    return F.softmax(x,-1)
  
actor = ActorNetwork()
actor.load_state_dict(torch.load("demo/data/actor.pth"))

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


