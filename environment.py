from puzzle import easyBoard,solution
import torch,random,time,math,sys
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

            if cellColor != "white" and cellColor != "gold":
 
                self.cells[row][column].setText(str(value))
                color = ("transparent" if not true_value else "gold")
                    
                ubl = (3 if (column % 3 == 0 and column!= 0) else 0.5)
                ubt = (3 if (row % 3 == 0 and row!= 0) else 0.5)
                updatedStyle = [
                    "background-color: dark grey;"
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
                    normalStyle = [
                        "background-color: grey;",
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

def modifiables(tensor) -> list: 
    # returns modifiables cells index of a board and a mask
    modlist = []
    for i,x in enumerate(tensor):
        for y in range(9): 
            if x[y] == 0: 
                modlist.append((i,y))
    return modlist


#### TODO : remove that pytorch layer and only use numpy
def region(index:tuple|list,board: torch.Tensor | np.ndarray): 
    if isinstance(board,np.ndarray): 
        board = torch.from_numpy(board).detach().clone()
    # returns the region (row,column,block) of a cell
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


#### TODO : remove that pytorch layer and only use numpy
class reward_function: # domain propagation
    def __init__(self,state = None,modCells:list = None):
        self.board = torch.tensor(state).clone()
        self.solution = solution
        self.modCells = modCells
        self.maxStep = len(modCells)*3
         
    def domain(self,idx:tuple|list) -> list :
        Region = region(idx,self.board )
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
        if isinstance(self.board,(np.ndarray,torch.Tensor)):
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

app = QApplication.instance()
if app is None:
    app = QApplication([])

register( id="sudoku", entry_point="__main__:environment")

class environment(gym.Env): 

    puzzle = easyBoard
    metadata = {"render_modes": ["human"]}   

    def __init__(self,render_mode = None):
        super().__init__()
        self.gui = Gui()

        self.action = None
        self.trueaction = False
        self.action_space = spaces.Tuple(
            (
                spaces.Discrete(9,None,0),
                spaces.Discrete(9,None,0),
                spaces.Discrete(9,None,1)
            )
        )
        self.observation_space = spaces.Box(1,9,(9,9),dtype=float)

        self.state = self.puzzle
        self.modif_cells : list = modifiables(easyBoard)
        self.rewardfn = reward_function
        
        self.render_mode = render_mode
        self.timer = QTimer()
        
    def reset(self,*kwargs) -> np.array :
        super().reset(seed=None)
        self.state = self.puzzle
        return np.array(self.state),{}

    def step(self,action):  
        self.action = action
        x,y,value = self.action 
        self.state[x][y] = value
        solvable = self.rewardfn(self.state,self.modif_cells).isSolvable()

        if solvable:
            reward = 10
            self.state[x][y] = value
            if (x,y) in self.modif_cells:
                self.modif_cells.remove(action[:2])
                self.trueaction = True

        elif not solvable:
            reward = -10
            self.trueaction = False
                
        info = {}
        done = False
        return np.array(self.state),reward,False,done,info
    
    def render(self):
        if self.render_mode == "human":
            self.state = self.gui.updated(self.action,self.trueaction)
            self.gui.show()
            app.processEvents()
            time.sleep(0.2)
        else : 
            sys.exit("render_mode attribute should be set to \"human\"")
         

if __name__=="__main__":
    
    easyBoard[6][2] = 9
    mod = modifiables(easyBoard)
    r = reward_function(easyBoard,mod).isSolvable()
    print(r)

    
"""
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

                    
modifiableCells = modi(easyBoard)


"""