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

 

x = torch.rand(size=(1,9,9))
 
@torch.no_grad()
def weights_init(w):
  if isinstance(w,(nn.Conv2d,nn.LazyConv2d,nn.LazyLinear)):
    nn.init.kaiming_uniform(w.weight,mode="fan_in",nonlinearity="relu")
    if w.bias is not None : nn.init.zeros_(w.bias)

def networkInit(network : nn.Module):
  network.forward(x)
  network.apply(weights_init)
  return network

class Mask: 
  # This will alter the softmax distribution so value in [x,y,value] != 0 
  def __init__(self):
    self.newValue = -float("inf")

  def apply(self,tensor : torch.FloatTensor):
    self.mask = torch.zeros_like(tensor,dtype=torch.bool)
    self.mask[-1,-1,0] = True
    tensor = tensor.masked_fill(mask=self.mask,value=self.newValue)
    return tensor


class ActorNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.size = 81
    self.outputShape = 27 # 3*9 = 27 haha
    self.outputReshaped = (3,9)
    self.mask = Mask()

    self.input_layer = nn.LazyLinear(81)
    self.flat = nn.Flatten()
    self.dense_one = nn.LazyLinear(self.size)
    self.dense_two = nn.LazyLinear(self.size)
    self.output = nn.LazyLinear(self.outputShape)

  def forward(self,x):
    x = self.flat(x)
    x = F.relu(self.input_layer(x))
    x = F.relu(self.dense_one(x))
    x = F.relu(self.dense_two(x))
    x = F.relu(self.output(x))
    x = torch.unflatten(x,-1,(self.outputReshaped))
    x = self.mask.apply(x)
    return F.softmax(x,-1)
         
Actor = networkInit(ActorNetwork())
Actor.load_state_dict(torch.load("./actor_1M.pth"),strict=False)

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
 
