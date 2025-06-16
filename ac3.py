from puzzle import easyBoard
import torch


easyBoard = easyBoard.to(torch.int16).tolist()

class arc:
    def __init__(self):
        pass
    
    def domain_definition(self,board):
        domain = list(range(1,10))
        data = {}
        for line in board :
            for item in line:
                data.update({item:domain})
        return data.keys()


s = arc()
print(s.domain_definition(easyBoard))