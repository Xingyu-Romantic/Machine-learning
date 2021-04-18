import numpy as np


class perceptron:
    def __init__(self):
        self.M = {} # 储存误分类点
        self.parameters = {} # parameters = {'W': W, 'X': X, 'b' : b}

    def sign(self.x): 
        return 1 if x>=0 else -1

    def norm(self, X, type = 'L2'):
        '''
        type = L1 or L2
            L1: 向量中各个元素绝对值之和
            L2：向量所有元素的平方和的开平方
        '''
        if type == 'L1':
            return np.sum(np.abs(X))
        elif type == 'L2':
            return np.sqrt(np.sum(np.square(X)))
        else:
            print('未知定义, L1 or L2')
            return


    def loss(W, M, Y):
        '''
        误分类点到超平面的距离，不考虑 1 / norm(w)
        parameters = {'W': W, 'X': X, 'b' : b}
        '''
        W, X, b = self.parameters['W'], self.parameters['X'], self.parameters['b']
        assert W.shape == X.shape == Y.shape == b.shape
        return -1 * np.sum(np.maximum(Y * (W * X + b), 0))

    '''
    W = np.array([1, 2, 5])
    X = np.array([-1, 3, 6])
    b = np.array([0, 0, 0])
    Y = np.array([1, 1, 1])
    parameters = {'W': W, 'X': X, 'b' : b}
    print(W * X.T)
    print(loss(parameters, Y))
    '''

    def fit(self, X, Y, lr = 1e-6):
        self.X = X
        self.Y = Y
        self.W = np.random.rand(X.shape)
        
        for 

