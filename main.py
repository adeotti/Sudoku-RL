import numpy as np
from dataclasses import dataclass
import math,sys
from puzzle import easyBoard,solution
import torch
np.random.seed(42)

from PySide6.QtWidgets import QApplication 
from PySide6.QtCore import QTimer
 

@dataclass(frozen=True)
class Board_specs:
  size: tuple = (9,9)
  low: int = 1
  high: int = 9

class Game:
    def __init__(self,action = None):
        self.board = easyBoard.to(int).numpy()
        self.action = action
        self.reward = 0
        self.done = np.array_equal(easyBoard,solution)

        self.modifiableCells = []
        
        for i,x in enumerate(self.board):
            for y in range(Board_specs.high): 
                if x[y] == 0: 
                    self.modifiableCells.append((i,y))    

    def Updated_board(self):
        if self.action is not None:
            row,column,value = self.action
            if (row,column) in self.modifiableCells:

                x = self.board[row].tolist()
                x.pop(column)
            
                y = [element[column].item() for element in self.board]
                y.pop(row)
                    
                n = int(math.sqrt(Board_specs.high))
                ix,iy = (self.action[0]//n)* n , (self.action[1]//n)* n
                region = torch.flatten(torch.from_numpy(self.board[ix:ix+n , iy:iy+n])).tolist()

                local_row = row - ix
                local_col = column - iy
                action_index = local_row * n + local_col
                region_ = [num for idx, num in enumerate(region) if idx != action_index]

                sector = [x,y,region_]
                sector = [item for sublist in sector for item in sublist]
                sector_ = [element for element in sector if element !=0] # filtered the zeros

                if not value in sector_:
                    self.board[row][column] = value
                    self.reward +=10

                    if self.done :
                        self.reward+= 20
                    return self.board,self.reward,self.done
                
                else :
                    self.reward -= 2
                return self.board,self.reward,self.done

            else:
                self.reward -=2
        return self.board,self.reward,self.done
    
 

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
        self.game = Game(self.action)
        self.updatedBoard,self.reward,self.done = self.game.Updated_board()

        self.gui = Gui()

        self.size = Board_specs.high
        self.action_space = spaces.Tuple((spaces.Discrete(Board_specs.high),                        # x
                                          spaces.Discrete(Board_specs.high),                        # y
                                          spaces.Discrete(Board_specs.high,start=Board_specs.low))) # value, start = 1
        self.observation_space = spaces.Box(low=Board_specs.low,high=Board_specs.high,
                                            shape=Board_specs.size,dtype=float)
        
    def reset(self,seed = None) :
        super().reset(seed=seed)
        return np.float16(self.updatedBoard),{}
        
    def step(self,action):

        self.game = Game(self.action)
        self.updated,self.reward,self.done = Game(self.action).Updated_board()
       
        info = {}
        self.action = action
        return  np.float16(self.updatedBoard),self.reward,self.done,False,info
    
    def render(self):
        self.gui.updated(self.action)
        self.gui.show()
   
        
if __name__=="__main__":
    """
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
                self.timer.start(200)
                app.exec()
            else:
                while self.counter < self.episodes:
                    self.stepComputing()
                    print(self.action)
                    self.counter+=1
          
    t = Test(render=True,episodes=100)
    t.run()
    """


t = ENVI()
 
t.render()