import numpy as np
from dataclasses import dataclass
import math,sys
np.random.seed(42)

from PySide6 import QtCore
from PySide6.QtWidgets import QApplication, QWidget,QGridLayout,QLineEdit
from PySide6.QtGui import QIcon 
from PySide6.QtCore import QTimer
from PySide6.QtGui import QCloseEvent

@dataclass(frozen=True)
class Board_specs:
  size: tuple = (9,9)
  low: int = 1
  high: int = 9

def Board(specs=Board_specs):
  matrix = np.random.randint(specs.low,specs.high,specs.size)
  return matrix

def regionSplit(matrix):
  # Convert the matrix into region size (3,3) 
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
    subgrid = [np.concatenate(line) for line in subgrid]
    return subgrid 

class Game:
    def __init__(self,action = None):
        self.board = Board()
        self.action = action
        
    def Updated_board(self):
        if self.action is not None:
            self.action = self.action[0] if not len(self.action)== 3 else self.action
            row,column,value = self.action
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
        cls.y = [line for line in cls.upd_matrix.T]
        cls.subgrid = regionSplit(cls.upd_matrix)
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
    # Reward function
    def __init__(self,instance):
        Util.Utilities(instance)
        self.action = Util.action
        self.matrix = Util.upd_matrix
        self.y_list = Util.y_list
        self.reward = 0
        if not len(self.action)==3: action = action[0]
        assert len(self.action)==3
       
    def rewardReturns(self):
        if self.action is None : 
            sys.exit(f"{self.__class__.__name__} : Action is None")
        # x
        x = self.matrix[self.action[0]]; assert len(x) == Board_specs.high
        self.reward += 9 if (Util.len_dif(x) == 0) else -Util.len_dif(x)
        # y
        y_index = self.action[1]
        y = self.y_list[y_index]; assert len(y) == Board_specs.high
        self.reward += 9 if (Util.len_dif(y) == 0) else -Util.len_dif(y)
        # region
        n = int(math.sqrt(Board_specs.high))
        ix,iy = (self.action[0]//n)* n , (self.action[1]//n)* n
        region = np.concatenate(self.matrix[ix:ix+n , iy:iy+n])
        assert len(region) == Board_specs.high
        self.reward += 9 if (Util.len_dif(region) == 0) else  -Util.len_dif(region)
        return self.reward

def gameEnd(instance):
    Util.Utilities(instance)
    matrix = Util.upd_matrix
    y_list = Util.y_list
    subgrid = Util.subgrid
    all_uniquerow = all(Util.unique(line) for line in matrix)
    all_uniquecol = all(Util.unique(line) for line in y_list)
    all_unique_region = all(Util.unique(line) for line in subgrid)
    return all_uniquerow,all_uniquecol,all_unique_region


import gymnasium as gym
import gymnasium.spaces as spaces
from gymnasium.envs.registration import register

from gui import Gui
app = QApplication.instance()
if app is None:
    app = QApplication([])

register(
    id="sudoku",
    entry_point="__main__:ENVI"
)

class ENVI(gym.Env):
    def __init__(self):
        super().__init__()
        self.action = None
        self.game = Game(None)
        self.updated,_ = self.game.Updated_board()

        self.gui = Gui()

        self.size = Board_specs.high
        self.action_space = spaces.Tuple((spaces.Discrete(Board_specs.high),                        # x
                                          spaces.Discrete(Board_specs.high),                        # y
                                          spaces.Discrete(Board_specs.high,start=Board_specs.low))) # value, start = 1
        self.observation_space = spaces.Box(low=Board_specs.low,high=Board_specs.high,
                                            shape=Board_specs.size,dtype=float)
        
    def reset(self,seed = None) :
        super().reset(seed=seed)
        return np.float16(self.updated) ,{}
        
    def step(self,action):
        self.game = Game(action)
        self.updated,_ = Game(action).Updated_board()
       
        reward_base = rewardFunction(self.game).rewardReturns()
        heuristic = boardConflicts(self.game).count()
        reward = reward_base + heuristic
        terminated_list = gameEnd(self.game)
        terminated = (True if all(terminated_list) else False)

        info = {
            "conflicts" : heuristic
        }
        self.action = action
        return  np.float16(self.updated),reward,terminated,False,info
    
    def render(self):
        self.gui.updated(self.action)
        self.gui.show()
   
        
if __name__=="__main__":
    class Test:
        def __init__(self, render:bool, episodes:int):
            
            self.timer = QTimer()
            self.render = render

            self.environmentClass = ENVI()
            self.terminated = False
            self.counter = 0
            self.episodes = episodes
        
        def stepComputing(self):
            self.environmentClass.reset()
            self.action = self.environmentClass.action_space.sample()
            observation, reward, terminated, truncated, info = self.environmentClass.step(self.action)
            return  terminated
            
        def guiRender(self):
            if self.counter < self.episodes:
                self.environmentClass.render()
                self.stepComputing()
                self.counter += 1
            else:
                self.timer.stop()
                sys.exit() 
        
        def run(self):
            if self.render:
                self.timer.timeout.connect(self.guiRender)
                self.timer.start(100)
                app.exec()
            else:
                while self.counter < self.episodes:
                    self.stepComputing()
                    print(self.action)
                    self.counter+=1
                    
    t = Test(render=False,episodes=10)
    t.run()

