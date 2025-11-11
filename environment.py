from puzzle import easyBoard,solution
import torch,random,time,math,sys
from torch import Tensor
import numpy as np

from PySide6 import QtCore,QtGui
from PySide6.QtWidgets import QApplication, QWidget,QGridLayout,QLineEdit
from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon 
from puzzle import easyBoard,solution

import gymnasium as gym
import gymnasium.spaces as spaces
from gymnasium.envs.registration import register

import PySide6

easyBoard = easyBoard.to(int).numpy()

class Gui(QWidget):
    def __init__(self ):
        super().__init__()
        self.setWindowTitle("Sudoku")
        self.setMaximumSize(20,20)
        self.setWindowIcon(QIcon("icon.png"))
        self.game = easyBoard
        self.grid = QGridLayout(self)
        self.grid.setSpacing(0)
        self.size = 9
        self.cells = [[QLineEdit(self) for _ in range(self.size)] for _ in range (self.size)] 
        for line in self.game :
            for x in range(self.size):
                for y in range(self.size):
                    self.cells[x][y].setFixedSize(40,40)
                    self.cells[x][y].setReadOnly(True)
                    number = str(easyBoard[x][y])
                    self.cells[x][y].setText(number)
                    self.bl = (3 if (y%3 == 0 and y!= 0) else 0.5) # what is bl,bt ? 
                    self.bt = (3 if (x%3 == 0 and x!= 0) else 0.5)
                    self.color = ("transparent" if int(self.cells[x][y].text()) == 0 else "white")
                    self.cellStyle = [
                        "background-color:grey;"
                        f"border-left:{self.bl}px solid black;"
                        f"border-top: {self.bt}px solid black;"
                        "border-right: 1px solid black;"
                        "border-bottom: 1px solid black;"
                        f"color: {self.color};"
                        "font-weight: None;"
                        "font-size: 20px"
                    ]
                    self.cells[x][y].setStyleSheet("".join(self.cellStyle))
                    self.cells[x][y].setAlignment(QtCore.Qt.AlignCenter)
                    self.grid.addWidget(self.cells[x][y],x,y)

    def updated(self,action = None,true_value : bool = False) -> list[list[int]]: 
        if action is not None:
            self.action = action
            assert isinstance(action,(tuple,list,np.ndarray,torch.Tensor))

            if not len(action) == 3:
                action = action[0] 
            assert len(action) == 3

            row,column,value = action
            # Checking the cell color, not every cell should be modifiable
            styleList = self.cells[row][column].styleSheet().split(";")
            if len(styleList) != 8 : 
                del styleList[-1]
            styleDict = {k.strip() : v.strip() for k,v in (element.split(":") for element in styleList)}
            cellColor = styleDict["color"]

            if cellColor != "white" and cellColor != "black":
                self.cells[row][column].setText(str(value))
                color = ("transparent" if not true_value else "black")
                ubl = (3 if (column % 3 == 0 and column!= 0) else 0.5)
                ubt = (3 if (row % 3 == 0 and row!= 0) else 0.5)
                updatedStyle = [
                    "background-color:dark grey;"
                    f"border-left:{ubl}px solid black;"
                    f"border-top: {ubt}px solid black;"
                    "border-right: 1px solid black;"
                    "border-bottom: 1px solid black;"
                    f"color: {color};"
                    "font-weight: None;"
                    "font-size: 20px"
                ]
                self.cells[row][column].setStyleSheet("".join(updatedStyle)) # Update the cell color flash

                def reset_style():
                    background = "orange" if color == "black" else "grey"
                    normalStyle = [
                        f"background-color:{background};",
                        f"border-left:{ubl}px solid black;",
                        f"border-top: {ubt}px solid black;",
                        "border-right: 1px solid black;",
                        "border-bottom: 1px solid black;",
                        f"color: {color};",
                        "font-weight: None;",
                        "font-size: 20px;"
                    ]
                    self.cells[row][column].setStyleSheet("".join(normalStyle)) 

                QTimer.singleShot(20, reset_style)  # Delay in milliseconds
                
                styleList = self.cells[row][column].styleSheet().split(";")
                styleDict = {k.strip() : v.strip() for k,v in (element.split(":") for element in styleList)}
                cellColor = styleDict["color"]
        
        list_text = [] 
        for rw in self.cells :
            for cells in rw:
                list_text.append(cells.text())   
        list_text = [int(element) for element in list_text]
        matrix = np.array([list_text],dtype=float).reshape(9,9)
        return matrix

#@torch.compile(mode="reduce-overhead",fullgraph=True)
def region_fn(index:list,board:Tensor): # returns the region (row ∪ column ∪ block) of a cells 
    x,y = index

    xlist = self.board[x]
    xlist = torch.cat((xlist[:y],xlist[y+1:]))
    ylist = self.board[:,y]
    ylist = torch.cat((ylist[:x],ylist[x+1:]))
    
    n = int(torch.tensor(9).sqrt())
    ix,iy = (x//n)* n , (y//n)* n
    block = torch.flatten(board[ix:ix+n , iy:iy+n])
    local_row = x - ix
    local_col = y - iy
    action_index = local_row * self.n + local_col
    block_ = torch.cat([block[:action_index], block[action_index+1:]]) 
    return   torch.cat([xlist,ylist,block_])


class reward_cls: 
    def __init__(self,board:Tensor,action:list,region):
        self.board = torch.tensor(board).clone()
        self.action = action
        self.x,self.y,self.target = self.action
        self.reward = 0
        self.mask = (self.board!=0)
        self.region = region
                           
    def reward_fn(self):
        if self.mask[self.x,self.y]:
            return 0.0
        #self.region = region_fn((self.x,self.y), self.board) # region comp 1   
        self.conflicts = (self.board == 0).sum().tolist()  
        self.unique = not torch.any(self.region==self.target).item()
        if self.unique:
            self.reward = 1 + (self.conflicts*0.1)
        else:
            self.reward = - (1 + self.conflicts*0.1)
        return round(self.reward,2)
           

class constrain_propagation: # one step constrain propagation 
    def __init__(self,region):
        self.region = region


app = QApplication.instance()
if app is None:
    app = QApplication([])

register( id="sudoku", entry_point="__main__:environment")


class environment(gym.Env): 
    puzzle = easyBoard
    metadata = {"render_modes": ["human"],"render_fps":4}   
    def __init__(self,render_mode = None):
        super().__init__()
        self.gui = Gui()
        self.action = None
        self.true_action = False
        self.action_space = spaces.Tuple(
            (
            spaces.Discrete(9,None,0),
            spaces.Discrete(9,None,0),
            spaces.Discrete(9,None,1)
            )
        )
        self.observation_space = spaces.Box(0,9,(9,9),dtype=np.int32)

        self.state = self.puzzle
        self.clone = self.state.copy()
        self.modif_cells : list = modifiables(easyBoard)
        self.region = region_fn
        self.constrain_prop = constrain_propagation
        self.rewardfn = reward_cls 
        self.render_mode = render_mode
                
    def reset(self,seed=None, options=None) -> np.array :
        super().reset(seed=seed)
        self.state = self.puzzle
        return np.array(self.state,dtype=np.int32),{}

    def step(self,action):   
        self.action = action
        x,y,value = self.action 
        self.clone[x][y] = value
        region = self.region((x,y),self.clone)

        reward = self.rewardfn(self.state,action,region).reward_fn()
        constrain = self.constrain_prop(region)
     
        if reward > 0:
            self.state[x][y] = value
            self.modif_cells.remove(action[:2])
            self.true_action = True
            self.clone = self.state
        else:
            self.true_action = False
        info = {}
        done = False
        truncated = False
        return np.array(self.state,dtype=np.int32),reward,truncated,done,info

    def render(self):
        if self.render_mode == "human":
            self.state = self.gui.updated(self.action,self.true_action)
            self.gui.show()
            app.processEvents()
            time.sleep(0.2)
        else :
            sys.exit("render_mode attribute should be set to \"human\"")

if __name__=="__main__":
    env = gym.make("sudoku",render_mode = "human")
    env.reset()
    for n in range(1000):
        obs,reward,trunc,done,info = env.step(env.action_space.sample())
        env.render()



