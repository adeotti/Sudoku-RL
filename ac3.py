from puzzle import easyBoard
import torch,sys


easyBoard = easyBoard.to(torch.int16).tolist()

class arc:
    def __init__(self,board):
        self.board = board
        self.domain : dict = self.domain_definition(self.board)
        self.k_matrix : list = self.keys_matrix(self.domain)
    
    def domain_definition(self,board):
        data = {}
        for x,line in enumerate(board):
            for y,item in enumerate(line):
                domain = list(range(1,10))
                if item in domain:
                    domain.remove(item)
                data.update({(x,y) : domain})
        return data
    
    def keys_matrix(self,data : dict):
        matrix = []
        n = 0
        d = list(data.keys())
        for e in range(9):
            row = d[n : n+9]
            matrix.append(row)
            n+= 9
        return matrix

    
    def arc_definition(self):
        return None

    def matrix_domain(self,board : list[list[int]]) -> list[list[object]]:
        """
        Returns a 9x9 matrix of node instances, each holding a domain of values in {1,9}.
        Values already present in the input board are removed from the corresponding domains.
        """
        class node:
            def __init__(self):
                self.domain = list(range(1,10))
        
        matrix1 = [[node() for _ in range(9)] for _ in range(9)]
        matrix2 = board

        for x1,x2 in zip(matrix1,matrix2):
            for nod,value in zip(x1,x2):
                if value != 0 and value in nod.domain:
                    nod.domain.remove(value)
        
        return matrix1
        

s = arc(easyBoard)
s.matrix_domain(easyBoard)
 