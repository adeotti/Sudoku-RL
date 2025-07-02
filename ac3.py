from puzzle import easyBoard
from collections import deque

class node:
    def __init__(self):
        self.value : int = 0
        self.indice : tuple = None
        self.domain = list(range(1,10))
        self.region : int = None

class arc_3:
    def __init__(self,board : list):
        self.board = board
        self.matrix = self.matrix_domain(self.board)  # work with this matrix
        self.colums = self.get_colums(self.matrix)
        self.regions = self.get_regions(self.matrix)
        arcs = self.arcs_merger()
       
    def matrix_domain(self,board : list[list[int]]) -> list[list[node]]:
        matrix1 = [[node() for _ in range(9)] for _ in range(9)]
        matrix2 = board
        for x1,x2 in zip(matrix1,matrix2):
            for nod,value in zip(x1,x2):
                if value != 0 and value in nod.domain:
                    nod.value = value
                    nod.domain.remove(value)
        
        for x in range(9): # index and region attributes
            for y in range(9):
                matrix1[x][y].indice = (x,y)
                matrix1[x][y].region = self.get_region_id(*(x,y))

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
    
    def get_region_id(self,x,y):
        return  (x // 3) * 3 + (y // 3)
        
    def arcs_definition(self,matrix : list[list[node]]):
        arcs = set()
        for line in matrix:
            for i in range(len(line)):
                for j in range(len(line)):
                    if i != j :
                        arcs.add((line[i], line[j]))
        return arcs
    
    def create_arcs(self):
        arcs_row = self.arcs_definition(self.matrix)
        arcs_col = self.arcs_definition(self.colums)
        arcs_regions = self.arcs_definition(self.regions)
        return arcs_row,arcs_col,arcs_regions
    
    def arcs_merger(self):
        arcs_x,arcs_y,arcs_regions = self.create_arcs()
        assert len(arcs_x) == len(arcs_y) == len(arcs_regions) == 648
        final_arcs = arcs_x | arcs_y | arcs_regions
        return final_arcs
    
    def revise(self,node1:node,node2:node):
        revised = False
        for x in node1.domain.copy():
            if not any(x!=y for y in node2.domain):
                node1.domain.remove(x)
                revised = True
        return revised
    
    def main(self):
        matrix = self.matrix
        new_matrix = []
        queue = deque(self.arcs_merger())
        while queue:
            node1,node2 = deque.popleft()
            if self.revise(node1,node2):
                pass

       





solver = arc_3(easyBoard)
print(solver.matrix[0][0].domain)
print(solver.matrix[4][5].region)
print(solver.matrix[2][7].value)
print(solver.matrix[8][8].indice)

 


 
 