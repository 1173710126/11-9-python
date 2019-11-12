import numpy as np

class MultivariateGaussian(object):
    # initialize n_dim、 mean、 covariance matrix
    def __init__(self, mean, cov):
        self.n_dim = mean.shape[0]
        self.mean = mean
        self.cov = cov
        self.precision = np.linalg.inv(self.cov)
    
    # 计算x在该多维高斯分下出现的概率, 注意：概率密度可以大于1，是分布函数一定<=1
    def prob(self, x):
        diff = x - self.mean
        exponent = -1/2 * np.matmul(np.matmul(diff.T, self.precision), diff)
        
        det = np.linalg.det(self.cov)
        factor = 1 / np.sqrt(det * ((2 * np.pi) ** self.n_dim)) # np.pi与math.pi相同, 都是float类型

        return  factor * np.exp(exponent)