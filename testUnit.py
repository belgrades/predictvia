import unittest
from  classifier import RandomForest_scikit
import pickle

class TestStringMethods(unittest.TestCase):

  def test_upper(self):
      rf = RandomForest_scikit()
      file = open("models.obj",'r')
      models = pickle.load(file)
      samples = models[0]
      responses= models[1]
      rf.train(samples,responses)
      rf.test(samples,responses,True)




if __name__ == '__main__':
    unittest.main()