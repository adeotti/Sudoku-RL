from environment import environment,reward_function,modifiables
from puzzle import easyBoard,solution
import unittest
import gymnasium as gym 


class Test(unittest.TestCase):

    env = gym.make("sudoku")
    env.reset()
    modcells = modifiables(easyBoard)

    def test_reward(self):
        _,reward,_,_,_ = self.env.step((6,2,9))
        self.assertLess(reward,0,"failed test reward")
    
    def test_solver_one(self):
        board_copy = easyBoard
        board_copy[6][2] = 8
        solver = reward_function(board_copy,self.modcells).isSolvable()
        self.assertTrue(solver)
    
    def test_solver_two(self):
        board_copy = easyBoard
        board_copy[6][2] = 9
        solver = reward_function(board_copy,self.modcells).isSolvable()
        self.assertFalse(solver)


if __name__ == "__main__":
    unittest.main()