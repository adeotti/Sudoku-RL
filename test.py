import sys
from main import ENVI 
import gymnasium, torch
from torch.distributions import Categorical
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
import torch
import torch.nn.functional as F
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
 

app = QApplication.instance()
if app is None:
    app = QApplication()

class Mask: 
  def __init__(self):
    self.newValue = -float("inf")

  def apply(self,tensor : torch.FloatTensor):
    self.mask = torch.zeros_like(tensor,dtype=torch.bool)
    self.mask[0,0,-1] = True
    self.mask[0,1,-1] = True
    self.mask[-1,-1,0] = True
    tensor = tensor.masked_fill(mask=self.mask,value=self.newValue)
    return tensor


class ActorNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.batchsize = 1
    self.action_dist = 30
    self.action_spec = (3,10)
    self.mask = Mask()

    self.input = nn.LazyLinear(9)
    self.conv1 = nn.LazyConv2d(self.batchsize,(3,3))
    self.conv2 = nn.LazyConv2d(self.batchsize,(3,3))
    self.conv3 = nn.LazyConv2d(self.batchsize,(3,3))
    self.conv4 = nn.LazyConv2d(self.batchsize,(3,3))
    self.output = nn.LazyLinear(self.action_dist)

  def forward(self,x:torch.Tensor):
    if not x.shape == torch.Size([1,9,9]) :
      x = x.unsqueeze(0)
    assert x.shape == torch.Size([1,9,9])

    x = self.conv1(x)
    x = F.relu(self.conv2(x))
    x = self.conv3(x)
    x = F.relu(self.conv4(x))
    x = torch.flatten(x,1,2)
    x = F.relu(self.output(x))
    x = torch.unflatten(x,-1,(self.action_spec))
    x = self.mask.apply(x)
    return F.softmax(x,-1)
         
ActorNetwork().forward(torch.rand((1,9,9),dtype=torch.float))
Actor = ActorNetwork()
Actor.load_state_dict(torch.load("./data/policies/policy4.pth"),strict=False)

env = gymnasium.make("sudoku")

def format_observation(env = env):
    observation = env.reset()[0]
    observation = torch.tensor(observation).float().unsqueeze(0)
    return observation

def action_generator(actor = Actor):
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
          print(f"{self.action}")
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

     
test = Test(render=True) # Setting render to false will lead to faster computing obviously.
test.run()
 
