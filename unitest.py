from environment import environment,reward_function,modifiables
from puzzle import easyBoard,solution
import unittest
import gymnasium as gym 
from ac3 import arc_3


class TestEnvironment(unittest.TestCase):
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


class TestAc3(unittest.TestCase):
    algo = arc_3()

    def test_node_properties(self):
        """
        test 
        print(solver.matrix[0][0].domain)
        print(solver.matrix[4][5].region)
        print(solver.matrix[2][7].value)
        print(solver.matrix[8][8].indice)
        """



if __name__ == "__main__":
    unittest.main()