import random 
import sys
import numpy as np
 
from PySide6 import QtCore
from PySide6.QtWidgets import QApplication, QWidget,QGridLayout,QLineEdit
from PySide6.QtGui import QIcon 
from puzzle import easy


class Gui(QWidget):
    def __init__(self ):
        super().__init__()

        self.setWindowTitle("Sudoku") 
        self.setMaximumSize(20,20)
        self.setWindowIcon(QIcon("icon.png"))

        self.game = easy
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
        self.cellStyle = ["background-color:grey;"
                            "border: 1px solid black;" 
                            "color: white"]
        # layout for cells 
        for line in self.game :
            for x in range(self.size):
                for y in range(self.size):
                    self.cells[x][y].setFixedSize(40,40)
                    self.cells[x][y].setReadOnly(True)
                    self.cells[x][y].setStyleSheet("".join(self.cellStyle))

                    if x % 3 == 0:
                        pass

                    """if (y!=0 and y % 3 == 0) :
                        self.cells[x][y].setStyleSheet(f"border-right: 2px ")"""
                    """if (  x % 3 == 0) :
                        self.cells[x][y].setStyleSheet(f" border-right: 2px  ") """
                    value = int(easy[x][y])
                    self.cells[x][y].setText("" if value == 0 else str(value))
                    self.cells[x][y].setAlignment(QtCore.Qt.AlignCenter)
                    self.grid.addWidget(self.cells[x][y],x,y)
                    
                    """if int(self.cells[x][y].text()) != 0:
                        print(self.cells[x][y].text())
                        #print(self.cells[x][y])"""
                        

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
                # TODO : assure that only cases containing a zero are modified 
                self.cells[row][column].setStyleSheet(f" border: 1px solid black; color: white;")
                #self.cells[row][column].setStyleSheet(f"background-color: grey;border: 1px solid black; color:black;")
                #else :
                #    pass
       
        list_text = [] 
        for rw in self.cells :
            for cells in rw:
                if not cells.text().isdigit():
                    cells.text() = 0
                print(cells.text() )
                 
                    
                #list_text.append(cells.text())   

        sys.exit()       
        list_text = [int(element) for element in list_text]

        matrix = [
            list_text[i:i+self.size] 
            for i in range(0,len(list_text),9)
        ]
        return matrix,self.action



app = QApplication([])

test = Gui()
test.updated((0,3,10))
test.show()
app.exec()

#{''.join([random.choice('0123456389ABCDEF') for _ in range(6)])}