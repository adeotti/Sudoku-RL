from torchrl.envs import EnvBase
from torchrl.data import BoundedTensorSpec,CompositeSpec
from tensordict import TensorDictBase,TensorDict
import torch
import sys
import math

easyBoard = torch.tensor([
    [0, 0, 0, 5, 3, 1, 0, 0, 0],
    [0, 0, 0, 0, 4, 0, 3, 0, 1],
    [1, 0, 0, 8, 0, 0, 0, 0, 0],
    [0, 0, 4, 0, 0, 5, 6, 0, 0],
    [0, 0, 3, 9, 0, 2, 1, 4, 0],
    [6, 1, 5, 0, 7, 0, 0, 9, 8],
    [0, 2, 0, 0, 9, 6, 0, 1, 0],
    [0, 5, 7, 2, 0, 8, 0, 0, 6],
    [0, 6, 1, 7, 5, 3, 0, 2, 4]])

solution = torch.tensor([
    [8, 4, 9, 5, 3, 1, 7, 6, 2],
    [5, 7, 2, 6, 4, 9, 3, 8, 1],
    [1, 3, 6, 8, 2, 7, 4, 5, 9],
    [2, 9, 4, 1, 8, 5, 6, 7, 3],
    [7, 8, 3, 9, 6, 2, 1, 4, 5],
    [6, 1, 5, 3, 7, 4, 2, 9, 8],
    [3, 2, 8, 7, 9, 5, 1, 6, 7],
    [4, 5, 7, 2, 1, 8, 9, 3, 6],
    [9, 6, 1, 7, 5, 3, 8, 2, 4]])

from dataclasses import dataclass

@dataclass(frozen=True)
class Board_specs:
  size: tuple = (9,9)
  low: int = 1
  high: int = 9

def Board():
  matrix = easyBoard
  return matrix

def Region_Split(matrix):
  # Extract the 3X3 regions
    ZERO = 0 ; CONST = 3 ; MAX_COLUMN_END = 12
    row_start = col_start = ZERO
    row_end = col_end = CONST
    subgrid = []
    while len(subgrid) < len(matrix) :
        subgrid.append([row[col_start:col_end] for row in matrix[row_start:row_end]])
        col_start += CONST
        col_end += CONST
        if col_end == MAX_COLUMN_END :
            col_start = ZERO
            col_end = CONST
            row_start += CONST
            row_end += CONST
    assert len(subgrid) == len(matrix)
    subgrid = [torch.concatenate(line) for line in subgrid]
    return subgrid

class Game:
    def __init__(self,action = None):
        self.board = Board()
        self.action = action

        self.modifiableCells = []
        
        for i,x in enumerate(self.board):
            for y in range(Board_specs.high):
                if x[y] == 0:
                    self.modifiableCells.append((i,y))            

    def Updated_board(self):
        if self.action is not None:
            
         
            #self.action = self.action#.tolist()[0] #tensor([[0, 1, 0]], dtype=torch.int32)
             
            
            row,column,value = self.action
            
            if (row,column) in self.modifiableCells:
                self.board[row][column] = value
                
        return self.board,self.action
    

class Util:
    @classmethod
    def Utilities(cls,gameinstance):
        if gameinstance is None :
            sys.exit(f"{cls.__name__} : No Game instance was provided")
        cls.upd_matrix,cls.action = gameinstance.Updated_board()

        # TODO : just transpose the matrix then ylist will be easy to implement
        cls.y_list = [[col[y_index] for col in cls.upd_matrix] for y_index in range(Board_specs.high)]
        cls.y = [line for line in torch.t(cls.upd_matrix)]
        cls.subgrid = Region_Split(cls.upd_matrix)
        cls.subgrid = [arr.tolist() for arr in cls.subgrid]

    @staticmethod
    def unique(lisst):
        return len(lisst) == len(set(lisst))

    @staticmethod
    def len_dif(lisst):
        return len(lisst) - len(set(lisst))


class boardConflicts: # Heuristic component

    previous_conflicts = 0

    def __init__(self,instance):
        Util.Utilities(instance)
        self.conflicts = 0
        self.previous_conflicts = 0
        self.reward_H = 0

    def count(self):
        for line in Util.upd_matrix:
            if not Util.unique(line) : self.conflicts += Util.len_dif(line)
        for col in Util.y_list:
            if not Util.unique(col) : self.conflicts += Util.len_dif(col)
        for submatrix in Util.subgrid:
            if not Util.unique(submatrix) : self.conflicts += Util.len_dif(submatrix)
        # H value
        if self.conflicts < self.previous_conflicts:
            self.reward_H = 2
        boardConflicts.previous_conflicts = self.conflicts
        return self.reward_H
    


class rewardFunction:
    def __init__(self,instance):
        Util.Utilities(instance)
        self.action = Util.action
        self.matrix = Util.upd_matrix
        self.y_list = Util.y_list
        self.reward = 0

    def rewardReturns(self):
        if self.action is None :
            sys.exit(f"{self.__class__.__name__} : Action is None")
        
        """if isinstance(self.action,torch.Tensor):
            sys.exit(self.action[0])
            self.action = self.action.tolist()[0]"""
            
        # x
      
        x = self.matrix[self.action[0]]#; assert len(x) == Board_specs.high
        self.reward += 9 if (Util.len_dif(x) == 0) else - Util.len_dif(x)
        # y
        y_index = self.action[1]
        y = self.y_list[y_index]; assert len(y) == Board_specs.high
        self.reward += 9 if (Util.len_dif(y) == 0) else -Util.len_dif(y)
        # region
        n = int(math.sqrt(Board_specs.high))
        ix,iy = (self.action[0]//n)* n , (self.action[1]//n)* n
        region = torch.flatten(self.matrix[ix:ix+n , iy:iy+n])
        assert len(region) == Board_specs.high
        self.reward += 9 if (Util.len_dif(region) == 0) else  -Util.len_dif(region)
        return self.reward


def gameEnd(instance):
    Util.Utilities(instance)
    matrix = Util.upd_matrix
    return torch.equal(solution,matrix)

from torchrl.envs import EnvBase
from torchrl.data import BoundedTensorSpec,CompositeSpec,OneHotDiscreteTensorSpec
from tensordict import TensorDictBase,TensorDict

class environment(EnvBase):
    def __init__(self):
        super().__init__()

        self.action = None
        self.game = Game(self.action)
        self.updatedBoard,_ = self.game.Updated_board()

        self.action_spec = BoundedTensorSpec(
            low=torch.tensor([0,0,1]), # the value on the last dim can't be equal to zero, range(1,9)
            high=torch.tensor([9,9,9]),
            shape=torch.Size([3,]),
            dtype=torch.int
        )

        self.observation_format = BoundedTensorSpec(
            low=1.0,
            high=9.0,
            shape=(easyBoard).unsqueeze(0).shape,
            dtype=torch.float32
        )
        self.observation_spec = CompositeSpec(observation = self.observation_format)

    def _step(self,tensordict) -> TensorDictBase :
        self.action = tensordict["action"].tolist()[0] # original shape -> tensor([[0, 1, 2]])
        #sys.exit(self.action)
        self.updated,_ = Game(self.action).Updated_board()

        self.game = Game(self.action)
        reward = rewardFunction(self.game).rewardReturns()

        output = TensorDict(
            {
                "observation" : self.updatedBoard.clone().detach().unsqueeze(0).float(),
                "reward" : torch.tensor(reward),
                "done" : gameEnd(self.game)
            }
        )
        return output

    def _reset(self,tensordict,**kwargs) -> TensorDictBase :
        output = TensorDict(
            {
                "observation" :  self.updatedBoard.clone().detach().unsqueeze(0).float()
                }
        )
        return output

    def _set_seed(self):
        pass

from tqdm import tqdm
from collections import deque

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical,Categorical

from tensordict.nn import TensorDictModule

from torchrl.modules import ValueOperator,ProbabilisticActor
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss

from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer,SamplerWithoutReplacement,LazyTensorStorage

from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hypers
l_rate = 0.01
sdg_momentum = 0.9

frames =  10             # number of steps
sub_frame = 5              # for the most inner loop of the training step
total_frames = 20      # maximum steps
epochs = 10

gamma = 0.80
lmbda = 0.99

env = environment()

dummy_observation = env._reset(None)["observation"] 
action_spec = tuple(env.action_spec.shape) # (3,9)
action_dist = env.action_spec.shape.numel() # 27
size = torch.flatten(dummy_observation).shape.numel() # 81


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


class ActorNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.size = size
    self.action_dist = 27
    self.action_spec = (3,9)

    self.input_layer = nn.LazyLinear(81)
    self.flat = nn.Flatten()
    self.dense_one = nn.LazyLinear(self.size)
    self.dense_two = nn.LazyLinear(self.size)
    self.output = nn.LazyLinear(self.action_dist)

  def forward(self,x):
    x = self.flat(x)
    x = F.relu(self.input_layer(x))
    x = F.relu(self.dense_one(x))
    x = F.relu(self.dense_two(x))
    x = F.relu(self.output(x))
    x = torch.unflatten(x,-1,(self.action_spec))
    x = F.softmax(x,-1)
    return x 

ActorNetwork().forward(dummy_observation)
Policy = TensorDictModule(module=ActorNetwork(), in_keys=["observation"],out_keys=["probs"])
PolicyModule = ProbabilisticActor(module=Policy ,spec=env.action_spec,in_keys=["probs"],
                       distribution_class = Categorical,return_log_prob = True)

Memory = ReplayBuffer(storage=LazyTensorStorage(max_size=frames),sampler=SamplerWithoutReplacement())
Collector = SyncDataCollector(create_env_fn=env,policy=PolicyModule,frames_per_batch=frames,total_frames=total_frames)


if __name__ == "__main__":
    Collector.rollout()
     
    #rolloutTest = environment()
    """maxSteps = 2
    container = []
    x = []
    for n in range(20):
        for element in rolloutTest.rollout(maxSteps)["next","observation"]:
            container.append(element)
        x.append(torch.equal(container[0],container[1]))

    print(x)""""""
    import torch.nn.functional as F
    from  torch.distributions import Categorical
    import time

    test = torch.rand((3,9))
    test[-1,0] = -float("inf")
    prob = F.softmax(test,-1)
    sample = Categorical(prob).sample()
    while sample[-1] !=0 :
        print(sample)
        time.sleep(0.2)
        sample = Categorical(prob).sample()"""
 
    
    
    
    """test = torch.rand(9)
    # Set the probability of sampling 0 to zero
    test[0] = -float('inf')  # This will become 0 after softmax
    sys.exit(test)
    prob = F.softmax(test, -1)
    sample = Categorical(prob).sample()
    while not sample == 0:
        print(sample)
        sample = Categorical(prob).sample()"""

  
    