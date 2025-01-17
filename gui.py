import random 
import numpy as np
 
from PySide6 import QtCore
from PySide6.QtWidgets import QApplication, QWidget,QGridLayout,QLineEdit
from PySide6.QtGui import QIcon 

class Gui(QWidget):
    def __init__(self ):
        super().__init__()

        self.setWindowTitle("Sudoku") 
        self.setMaximumSize(20,20)
        self.setWindowIcon(QIcon("icon.png"))

        self.grid = QGridLayout(self)
        self.grid.setSpacing(0)
        self.size = 9 
        self.action = None
        self.reward = 0
        self.reward_H = 0
        self.conflicts = 0
        self.previous_conflicts = 0

        self.cells = [
            [QLineEdit(self) for _ in range(self.size)] 
            for _ in range (self.size)
        ]
        # layout for cells 
        for x in range(self.size):
            for y in range(self.size):
                self.cells[x][y].setMaximumSize(23,23)
                self.cells[x][y].setReadOnly(True)
                self.cells[x][y].setStyleSheet(f"background-color: #{''.join([random.choice('0123456389ABCDEF') for _ in range(6)])};border: 1px solid balck; color: white;")
                self.cells[x][y].setText(str(random.randint(1,self.size)))
                self.cells[x][y].setAlignment(QtCore.Qt.AlignCenter)
                self.grid.addWidget(self.cells[x][y],x,y)

    def updated(self,action = None ) -> list[list[int]]: 
        #This method update the cells using the "action" parameter and return a matrix of the updated grid
        # The action is in the shape [x,y,value] or (x,y,value)
        if action is not None:
            self.action = action
            if not isinstance(action,(tuple,list,np.ndarray))  :
               print(f"action : {action} : {type(action)} should be a list, a tuple or a numpy array")
            else :
                if not len(action) == 3:
                    action = action[0] 
                assert len(action) == 3
                row,column,value = action
                self.cells[row][column].setText(str(value))
                self.cells[row][column].setStyleSheet(f"background-color: #{''.join([random.choice('0123456389ABCDEF') for _ in range(6)])};border: 1px solid balck; color: white;")

        list_text = [] 
        for rw in self.cells :
            for cells in rw:
                list_text.append(cells.text())          
        list_text = [int(element) for element in list_text]

        matrix = [
            list_text[i:i+self.size] 
            for i in range(0,len(list_text),9)
        ]
        return matrix,self.action