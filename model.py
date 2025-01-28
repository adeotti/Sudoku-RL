from main import ENVI
from tqdm import tqdm
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical

from tensordict.nn import TensorDictModule
from tensordict import TensorDict

from torchrl.modules import ValueOperator,ProbabilisticActor
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss
# environment 
from torchrl.envs import TransformedEnv,GymEnv,Compose,DoubleToFloat,UnsqueezeTransform
# data collection and manipulation
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer,SamplerWithoutReplacement,LazyTensorStorage
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")
 
import numpy as np

# hypers 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

l_rate = 0.01  
sdg_momentum = 0.9

frames =  1000               # number of steps
sub_frame = 500           # for the most inner loop of the training step
total_frames = 5000   # maximum steps
epochs = 10 

gamma = 0.80
lmbda = 0.99 

# Reproductibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# environnment
base = GymEnv("sudoku",device = device)
env = TransformedEnv(base,Compose(UnsqueezeTransform(in_keys=["observation"], unsqueeze_dim=0,allow_positive_dim=True),
                                  DoubleToFloat())
)

dummy_observation = env.reset()["observation"]
action_spec = tuple(env.action_spec.shape) # (3,9)
action_dist = env.action_spec.shape.numel() # 27
size = torch.flatten(dummy_observation).shape.numel() # 81

# Actor
class ActorNetwork(nn.Module):
  # In order to incorporate a mcts search into the system maybe the actor network should output
  # many probability distribution instead of just one 

  def __init__(self):
    super().__init__()
    self.size = size
    self.action_dist = action_dist
    self.action_spec = action_spec

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

@torch.no_grad()
def weights_init(w):
  if isinstance(w,(nn.Conv2d,nn.LazyConv2d,nn.LazyLinear)):
    nn.init.kaiming_uniform(w.weight,mode="fan_in",nonlinearity="relu")
    if w.bias is not None : nn.init.zeros_(w.bias) 

def Network_util(network : nn.Module):
  network.to(device)
  network.forward(dummy_observation)
  network.apply(weights_init)
  return network

Network_util(ActorNetwork())
#Actor =  Network_util(ActorNetwork())
Actor.load_state_dict(torch.load("100k_data/actor_100k.pth"))
PolicyModule = TensorDictModule(module=ActorNetwork(), in_keys=["observation"],out_keys=["probs"])

PolicyModule = ProbabilisticActor(
  module=PolicyModule, spec=env.action_spec, in_keys=["probs"] ,
  distribution_class=OneHotCategorical, return_log_prob=True
)

Memory = ReplayBuffer(storage=LazyTensorStorage(max_size=frames),sampler=SamplerWithoutReplacement())
Collector = SyncDataCollector(create_env_fn=env,policy=PolicyModule,frames_per_batch=frames,total_frames=total_frames)
Collector.rollout()

# Critic
class ValueNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.size = size
    self.action_dist = action_dist
    self.action_spec = action_spec

    self.input_layer = nn.LazyLinear(self.size)
    self.flat = nn.Flatten()
    self.dense_one = nn.Linear(self.size,self.size)
    self.dense_two = nn.Linear(self.size,self.size)
    self.output = nn.Linear(self.size,1)
    
  def forward(self,x):
    x = self.flat(x)
    x = F.relu(self.input_layer(x))
    x = F.relu(self.dense_one(x))
    x = F.relu(self.dense_two(x))
    return self.output(x)
 
Critic = Network_util(network=ValueNetwork())
ValueModule = ValueOperator(module= Critic,in_keys=["observation"])
Advantage = GAE(gamma=gamma,lmbda=lmbda,value_network=ValueModule,average_gae=True,device=device)

loss = ClipPPOLoss(actor_network=PolicyModule,critic_network=ValueModule)
optimizer = torch.optim.SGD(params=loss.parameters(),lr=l_rate,momentum=sdg_momentum )

class Training:
  def __init__(self):  
      self.policy = ActorNetwork()
      self.collector = Collector
      self.memory = Memory
      self.valuemodule = ValueModule
      self.advantage = Advantage
      self.lossfunction = loss
      self.optimizer = optimizer
      self.epochs = epochs

  def save_logs(self): 
    log_dir = "data/"
    self.writer = SummaryWriter(log_dir)
                  
  def save_weight(self): 
    path = "data/actor.pth"
    torch.save(self.policy.state_dict(),path)

  def train(self,start : bool = None):  
      if start:
          rewardHistory = deque(maxlen=10)
          bestReward = -5
          self.save_logs()
          for i,data_tensordict in tqdm(enumerate(self.collector),total = total_frames/frames):   
            
              for e in range(self.epochs):  
                  dat = self.advantage(data_tensordict)
                  dat["advantage"] = dat["advantage"].unsqueeze(-1)
                  self.memory.extend(dat)

                  for n in range(total_frames//sub_frame):
                      subdata = Memory.sample(sub_frame)
                      loss_val = self.lossfunction(subdata.to(device)) 
                      loss_value = (loss_val["loss_objective"] + loss_val["loss_critic"] + loss_val["loss_entropy"])
                      loss_value.backward()
                      self.optimizer.step()
                      self.optimizer.zero_grad()
          
              self.writer.add_scalar("main/batch_number",i)
              self.writer.add_scalar("main/Advantage",dat["advantage"][0].item())
              self.writer.add_scalar("main/Loss_sum",loss_value.item())
              self.writer.add_scalar("main/reward_advantage",dat["next"]["reward"][0].mean().item())
              self.writer.add_scalar("main/raw_reward",data_tensordict["next"]["reward"][0].mean().item())
              self.writer.add_scalar("loss/Loss_entropy",loss_val["loss_entropy"].item())
              self.writer.add_scalar("loss/Loss_critic",loss_val["loss_critic"].item())
              self.writer.add_scalar("loss/Loss_objective",loss_val["loss_objective"].item())

              currentReward = data_tensordict["next"]["reward"][0].mean()
              rewardHistory.append(currentReward)
              averageReward = sum(rewardHistory)/len(rewardHistory)

              if i % 10 == 0:
                 if averageReward > bestReward:
                    self.save_weight()
                    bestReward = averageReward

if __name__ == "__main__":
  Training().train(start=True)