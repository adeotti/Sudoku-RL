import sys
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
        #return self.terminated
        # You can normally return (self.terminated) to end the jobs when it value is True 
        # Reminder : a Sudoku game is consider as complete when all numbers on the x,y and region are unique 
        # self.terminated evaluate the uniqueness of each number on each x,y and region so if self.terminated = True,
        #   then the model have solved the game

    def guiRendering(self):
        while True:
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
            pass
            #while not self.stepComputing():

     




test = Test(render=True)
test.run()


