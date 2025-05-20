from puzzle import easyBoard,solution
import torch,random,time
import numpy as np

from PySide6 import QtCore,QtGui
from PySide6.QtWidgets import QApplication, QWidget,QGridLayout,QLineEdit
from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon 
from puzzle import easyBoard,solution

import gymnasium as gym
import gymnasium.spaces as spaces
from gymnasium.envs.registration import register

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
        self.cells = [
            [QLineEdit(self) for _ in range(self.size)] for _ in range (self.size)
        ]
        # layout for cells 
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
                    self.cellStyle = ["background-color:grey;"
                        f"border-left:{self.bl}px solid black;"
                        f"border-top: {self.bt}px solid black;"
                        "border-right: 1px solid black;"
                        "border-bottom: 1px solid black;"
                        f"color: {self.color};"
                        "font-weight: None;"
                        "font-size: 20px"]
                    self.cells[x][y].setStyleSheet("".join(self.cellStyle))
                    self.cells[x][y].setAlignment(QtCore.Qt.AlignCenter)
                    self.grid.addWidget(self.cells[x][y],x,y)

    def updated(self,action = None ) -> list[list[int]]: 
        if action is not None:
            self.action = action
            assert isinstance(action,(tuple,list,np.ndarray,torch.Tensor))
            if not len(action) == 3:
                action = action[0] 
            assert len(action) == 3
            row,column,value = action
            # Checking the cell color, not every cell should be modifiable
            styleList = self.cells[row][column].styleSheet().split(";")
            styleDict = {k.strip() : v.strip() for k,v in (element.split(":") for element in styleList)}
            cellColor = styleDict["color"]
            if cellColor != "white":
                self.cells[row][column].setText(str(value))
                self.ubl = (3 if (column%3 == 0 and column!= 0) else 0.5)
                self.ubt = (3 if (row%3 == 0 and row!= 0) else 0.5)
                updatedStyle = [
                    "background-color: grey;"
                    f"border-left:{self.ubl}px solid black;"
                    f"border-top: {self.ubt}px solid black;"
                    "border-right: 1px solid black;"
                    "border-bottom: 1px solid black;"
                    f"color: gold;"
                    "font-weight: None;"
                    "font-size: 20px"
                ]
                self.cells[row][column].setStyleSheet("".join(updatedStyle))
                # Update the cell color
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
 
 
app = QApplication.instance()
if app is None:
    app = QApplication([])

register( id="sudoku", entry_point="__main__:environment")

class environment(gym.Env): # TODO finish implementing reward function and step function
    def __init__(self):
        super().__init__()
        self.gui = Gui()
        self.action = None
        self.action_space = spaces.Tuple(
            (spaces.Discrete(9,None,0),spaces.Discrete(9,None,0),spaces.Discrete(9,None,1))
        )
        self.observation_space = spaces.Box(1,9,(9,9),dtype=float)
        self.modif_cells = None
        self.state = None
        self.timer = QTimer()

    def reset(self) -> np.array :
        super().reset(seed=12)
        self.state = self.gui.updated(None)
        return self.state,{}

    def step(self,action):  
        self.state = self.gui.updated(action)
        reward = 0
        info = {}
        done = False
        return np.array(self.state),reward,False,done,info

    def reward_function(self,state): 
        pass
    
    def render(self):
        self.gui.show()
        app.processEvents()
        time.sleep(0.5)
         

if __name__=="__main__":
    t = gym.make("sudoku")
    t.reset()
    
    for r in range(50):
        t.step((random.randint(0,8),random.randint(0,8),random.randint(1,9)))
        t.render()
    




"""
def modi(tensor) -> list: # returns modifiables cells index of a board and a mask
    modlist = []
    for i,x in enumerate(tensor):
        for y in range(9): 
            if x[y] == 0: 
                modlist.append((i,y))
    return modlist

def region(index:tuple|list,board: torch.Tensor): # return the region (row,column,block) of a cell
    x,y = index
    xlist = board[x].tolist()
    xlist.pop(y)

    ylist = [element[y].tolist() for element in board]
    ylist.pop(x)

    #block
    n = int(math.sqrt(9))
    ix,iy = (x//n)* n , (y//n)* n
    block = torch.flatten(board[ix:ix+n , iy:iy+n]).tolist()
    local_row = x - ix
    local_col = y - iy
    action_index = local_row * n + local_col
    block_ = [num for idx, num in enumerate(block) if idx != action_index]

    #output
    Region = [xlist,ylist,block_]
    Region = [item for sublist in Region for item in sublist]
    return Region

class solver: 
    def __init__(self,state:torch.Tensor,modCells:list):
        self.board = state
        self.solution = solution
        self.modCells = modCells
        self.maxStep = len(modCells)*3
         
    def domain(self,idx:tuple|list) -> list :
        Region = region(idx,self.board)
        Region = set([item for item in Region if item != 0]) 

        domain_ = set(range(1,10)) 
        TrueDomain = list(domain_ - Region)
        return TrueDomain
    
    def collector(self):
        queu = []
        for element in self.modCells:
            queu.append({element : self.domain(element)})
        return queu
    
    def isSolvable(self) -> bool: 
        count = 0
        while True:
            self.__init__(self.board,self.modCells)
            data = self.collector()
            for dictt in data:
                for k,v in dictt.items():
                    if len(v) == 1:
                        self.board[k] = v[0]
            count+=1
            if len(data) == 0:
                break
            else:
                if count > self.maxStep:
                    break
        diff = (self.board == solution)
        diff = (diff == True).sum().item()
        if diff == solution.numel(): # if all True cells = 81 :
            return True
        else:
            return False

            
modifiableCells = modi(easyBoard)



class Env:
    def __init__(self):
        self.modifiableCells = modifiableCells.copy()
        self.solution = solution
        self.state = easyBoard.clone()
    
    def reset(self):
        self.state  = easyBoard.clone()
        self.modifiableCells = modifiableCells.copy()

    def step(self,action : tuple|list):#,state:torch.Tensor):
        self.action = action
        x,y,value = self.action
        reward,conflicts = self.rewardFunction(action,self.state)
        if reward > 0:
            self.state[x][y] = value
            self.modifiableCells.remove((x,y))
        done = torch.equal(solution,self.state)  
        return [
                self.state, \
                torch.tensor([reward],dtype=torch.float),\
                torch.tensor([done]),  \
                torch.tensor([action]),\
                conflicts
                ]
           
    def rewardFunction(self,action:tuple|list,board:torch.Tensor):
        """"""
        This will call the solver method to check if the board is solvable after a cell is filled.
        This fill a copy of the given board so the result here does not affect the original state
        if the board is solvable then the index of the value (x,y) is removed from the list of modifiables cells
        """"""
        reward = 0
        x,y,value = action
        board = board.clone() 
        copyList = self.modifiableCells.copy()
        if not (x,y) in copyList:
            diff = (board == self.solution) 
            conflicts = (diff == False).sum().item() 
            return 0,conflicts
        board[x][y] = value
        conflicts = ((board == self.solution) == False).sum().to(float) 
        copyList.remove((x,y)) # remove (x,y) before passing it to Solver
        Solver = solver(board.clone(),copyList)
        if Solver.isSolvable():
            reward = (conflicts/2)*0.1 + 5 
        else:
            reward = -((conflicts/2)*0.1 + 5)
        return reward,conflicts.floor()


"""