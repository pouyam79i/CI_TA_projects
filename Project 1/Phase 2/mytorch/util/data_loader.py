import numpy as np
from mytorch import Tensor
from PIL import Image
from random import shuffle

# data count
TRAIN = 10000
TEST = 1000

class DataLoader:

    def __init__(self, train_addr:str, test_addr:str) -> None:
        # train
        self.train_addr = train_addr
        self.train_data = []
        # test
        self.test_addr = test_addr
        self.test_data = []

    def load(self):
        print("loading train...")
        for i in range(TRAIN):
            label = (int) (i/1000)
            index = (int) ((i%1000) + 1) 
            addr = self.train_addr + '/' + label.__str__() + ' (' + index.__str__() + ')' + '.jpg'
            img = Image.open(addr, mode='r')
            self.train_data.append((Tensor(np.array(img)), label))
        print("loading test...")
        for i in range(TEST):
            label = (int) (i/100)
            index = (int) ((i%100) + 1) 
            addr = self.test_addr + '/' + label.__str__() + ' (' + index.__str__() + ')' + '.jpg'
            img = Image.open(addr, mode='r')
            self.test_data.append((Tensor(np.array(img)), label))
        print('done!')

    def shuffle_train(self):
        shuffle(self.train_data)

    def shuffle_test(self):
        shuffle(self.test_data)
