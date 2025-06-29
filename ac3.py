from puzzle import easyBoard
import torch,sys
from collections import deque

class arc_3:
    def __init__(self,board : list):
        self.board = board
        self.matrix = self.matrix_domain(self.board)  # work with this matrix
        self.regions = self.get_regions(self.matrix)
        self.colums = self.get_colums(self.matrix)
    
    def matrix_domain(self,board : list[list[int]]) -> list[list[object]]:

        class node:
            def __init__(self):
                self.value : int = 0
                self.indice : tuple = None
                self.domain = list(range(1,10))
                
        matrix1 = [[node() for _ in range(9)] for _ in range(9)]
        matrix2 = board
        for x1,x2 in zip(matrix1,matrix2):
            for nod,value in zip(x1,x2):
                if value != 0 and value in nod.domain:
                    nod.value = value
                    nod.domain.remove(value)
        
        for x in range(9): # index attributes
            for y in range(9):
                matrix1[x][y].indice = (x,y)
        return matrix1
    
    def get_colums(self,board) -> list[list[int]]:
        transposed_board = []
        new_line = deque(maxlen=9)
        for v in range(9):
            for line in board:
                new_line.append(line[v])
            transposed_board.append(list(new_line))
        return transposed_board

    def get_regions(self,board):
        regions = []
        for block_row in range(0, 9, 3):  
            for block_col in range(0, 9, 3):   
                region = []
                for x in range(3):
                    for y in range(3):
                        region.append(board[block_row + x][block_col + y])
                regions.append(region)
        return regions
        
    def arcs_definition(self):
        pass
    
        
        

easyBoard = easyBoard.to(torch.int16).tolist()

s = arc_3(easyBoard)
print(s.matrix[8][0].indice)
 