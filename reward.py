from environment import easyBoard,solution
import torch,math,numba


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


class c_propagation:
    pass


class csp:
    pass


class backtracking:
    pass


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
        

if __name__ == "__main__":
    modifiableCells = modi(easyBoard)
    Solver = solver(torch.from_numpy(easyBoard),modifiableCells)
    import time

    start = time.time()
    print(Solver.isSolvable())
    end = time.time()
    print(end-start)

