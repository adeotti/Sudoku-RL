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
            self.action = self.action.int()#.tolist()[0] #tensor([[0, 1, 0]], dtype=torch.int32)
            #print(self.action[0])
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
      
        #sys.exit(self.matrix[self.action[0]])
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


class environment(EnvBase):
    def __init__(self):
        super().__init__()
        
        self.action = None
        self.game = Game(self.action)
        self.updatedBoard,_ = self.game.Updated_board()

        # specs
        self.action_spec = BoundedTensorSpec(
            low=torch.tensor([0,0,1]),
            high=torch.tensor([9,9,9]),
            shape=(3,),
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
        self.action = tensordict["action"]
        self.updated,_ = Game(self.action).Updated_board()

        self.game = Game(self.action)
        reward = rewardFunction(self.game).rewardReturns()

        output = TensorDict({"observation" : torch.tensor(self.updatedBoard).float(),
                             "reward" : torch.tensor(reward),
                             "done" : gameEnd(self.game)})
        return output
        
    def _reset(self,tensordict,**kwargs) -> TensorDictBase :
        output = TensorDict({"observation" : torch.tensor(self.updatedBoard).float() })
        return output

    def _set_seed(self):
        pass


if __name__ == "__main__":
     
    rolloutTest = environment()
    maxSteps = 2
    container = []
    x = []
    for n in range(20):
        for element in rolloutTest.rollout(maxSteps)["next","observation"]:
            container.append(element)
        x.append(torch.equal(container[0],container[1]))

    print(x)