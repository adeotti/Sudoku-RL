import sys
import numpy as np
 
from PySide6 import QtCore,QtGui
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

        self.number = None

        self.cells = [
            [QLineEdit(self) for _ in range(self.size)] 
            for _ in range (self.size)
        ]

        # layout for cells 
        for line in self.game :
            for x in range(self.size):
                for y in range(self.size):
                    self.cells[x][y].setFixedSize(40,40)
                    self.cells[x][y].setReadOnly(True)
                    
                    number = str(easy[x][y])
                    self.cells[x][y].setText(number)

                    self.bl = (3 if (y%3 == 0 and y!= 0) else 0.5)
                    self.bt = (3 if (x%3 == 0 and x!= 0) else 0.5)

                    self.color = ("transparent" if int(self.cells[x][y].text()) == 0 else "white" )

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
        # This method update the cells using the "action" parameter and return a matrix of the updated grid
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

                # Checking the cell color, not every cell should be modifiable
                styleList = self.cells[row][column].styleSheet().split(";")
                styleDict = {k.strip() : v.strip() for k,v in (element.split(":") for element in styleList)}
                cellColor = styleDict["color"]

                if cellColor != "white":
                    self.cells[row][column].setText(str(value))
            
                    self.ubl = (3 if (column%3 == 0 and column!= 0) else 0.5)
                    self.ubt = (3 if (row%3 == 0 and row!= 0) else 0.5)
                    
                    updatedStyle = ["background-color: grey;"
                        f"border-left:{self.ubl}px solid black;"
                        f"border-top: {self.ubt}px solid black;"
                        "border-right: 1px solid black;"
                        "border-bottom: 1px solid black;"
                        f"color: gold;"
                        "font-weight: None;"
                        "font-size: 20px"]
                    
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

        matrix = [
            list_text[i:i+self.size] 
            for i in range(0,len(list_text),9)
        ]
        return matrix,self.action


if __name__ == "__main__":
    app = QApplication([])
    test = Gui()
    test.show()
    app.exec()






    #print(self.cells[row][column].palette().color(QtGui.QPalette.Text).name())
    #self.color = self.cells[row][column].palette().color(QtGui.QPalette.Text).name()
       