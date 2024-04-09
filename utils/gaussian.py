from scipy.stats import norm

import os
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))
import numpy as np
from matplotlib import pyplot as plt
from utils import print_warning
import math
import random

epsilon = 1e-4
infinite = 1e5
min_sigma = 0.01

class Distribution():
    def __init__(self, mu=0, sigma=0) -> None:
        self.mu = mu
        self.sigma = max(sigma, -1*sigma)
        self.var = sigma*sigma

    def cdf(self, x):
        return 0
    
    def ppf(self, p):
        return 0
    
    def pdf(self, x):
        return 0
    
    def sample(self, num=1):
        uniform = Uniform()
        s = uniform.sample(num)
        return s

class JointDistribution2D():
    def __init__(self, distribution_x:Distribution, distribution_y:Distribution, rho) -> None:
        self.distribution_x = distribution_x
        self.distribution_y = distribution_y
        self.rho = rho

        self.mu_x = distribution_x.mu
        self.mu_y = distribution_y.mu

        self.sigma_x = distribution_x.sigma
        self.sigma_y = distribution_y.sigma

        self.covariance_xy = distribution_x.sigma*distribution_y.sigma*rho
        self.covariance_yx = self.covariance_xy


        self.cov = np.zeros([2,2], dtype=float)
        self.cov[0][0] = self.sigma_x*self.sigma_x
        self.cov[0][1] = self.covariance_xy
        self.cov[1][0] = self.covariance_yx
        self.cov[1][1] = self.sigma_y*self.sigma_y


    def sample(self, num=1):
        samples = conditional_correlation_sampling_2D(num,self.rho)

        samples[0, :] = self.distribution_x.ppf(samples[0, :])
        samples[1, :] = self.distribution_y.ppf(samples[1, :])

        return samples
    
class Uniform(Distribution):
    def __init__(self, lower=0, upper=1) -> None:
        self.mu = (lower+upper)/2
        self.var = ((upper-self.mu)**3 - (lower-self.mu)**3)/3
        self.sigma = math.sqrt(self.var)


    def cdf(self, x):
        if x < 0:
            return 0
        elif x>=0 and x<1:
            return x
        else:
            return 1
    
    def ppf(self, p):
        return p
    
    def pdf(self, x):
        if x>=0 and x<1:
            return 1
        else:
            return 0
    
    def sample(self, num=1, dead_zone = 0.01):
        x = random.random()
        # TODO: relate x to mu and sigma
        s = [max(min(x, 1-dead_zone),0+dead_zone) for _ in range(num)]
        return s
    
class Bernoulli(Distribution):
    def __init__(self, n=1, p=0.5) -> None:
        self.n = n
        self.p = p
        self.mu = n*p
        self.var = n*p*(1-p)
        self.sigma = math.sqrt(self.var)

    def cdf(self, x):
        if x<0:
            return 0
        elif x >= 0 and x < 1:
            return 1-self.p
        elif x>=1:
            return 1
        
    def ppf(self, p):
        if p <= 1-self.p:
            return 0
        else:
            return 1
        
    def pdf(self, p):
        if x == 0:
            return infinite*(1-p)
        elif x == 1:
            return infinite*(p)
        else:
            return 0
        
    def sample(self, num=1):
        s = []
        for _ in range(num):
            x = 0
            for __ in range(self.n):
                x += 1 if random.random() > self.p else 0
            s.append(x)
        return s

    
class Gaussian(Distribution):
    def __init__(self, mu, sigma):
        '''
        Sigma is not Variance!
        '''
        super(Gaussian, self).__init__()
        self.mu = mu
        self.sigma = max(min_sigma,max(sigma, -1*sigma))
        self.var = sigma*sigma

    def normalize(self, x):
        return (x-self.mu)/(self.sigma+epsilon)
    
    def anti_normalize(self, x):
        return (x*self.sigma+self.mu)

    def cdf(self, x):
        x = self.normalize(x)
        return norm.cdf(x)
    
    def ppf(self, p):
        x = norm.ppf(p)
        x = self.anti_normalize(x)
        return x
    
    def pdf(self, x):
        x = self.normalize(x)
        p = norm.pdf(x)
        return p
    
    # def sample(self, num=1):
    #     s = [self.ppf(random.random()) for _ in range(num)]
    #     return s

class BiGaussian(Distribution):
    def __init__(self, mu1, sigma1, mu2, sigma2, alpha=0.5):
        '''
        Sigma is not Variance!
        '''
        super(BiGaussian, self).__init__()

        self.mu1 = mu1
        self.sigma1 = max(min_sigma,max(sigma1, -1*sigma1))
        self.mu2 = mu2
        self.sigma2 = max(min_sigma,max(sigma2, -1*sigma2))
        self.alpha = alpha

        self.gaussian1 = Gaussian(mu1, sigma1)
        self.gaussian2 = Gaussian(mu2, sigma2)

        self.mu = mu1*alpha + mu2*(1-alpha)
        self.var = (self.sigma**2)*alpha + (self.sigma2**2)*(1-alpha) + alpha*(1-alpha)*(mu1-mu2)*(mu1-mu2)
        self.sigma = np.sqrt(self.var)

    def cdf(self, x):
        p = self.alpha*self.gaussian1.cdf(x)
        p += (1-self.alpha)*self.gaussian2.cdf(x)
        return p
    
    def ppf(self, p):
        bound = [-40,5e2]
        # bound.append(min(self.mu1-100*self.sigma1, self.mu2-100*self.sigma2))
        # bound.append(max(self.mu1+100*self.sigma1, self.mu2+100*self.sigma2))

        if type(p) == float or type(p) == int:
            x = solve(self.cdf, p, bound=bound)
        else:
            x = []
            for pi in p:
                xi = solve(self.cdf, pi, bound=bound)
                x.append(xi)
        return x
    
    def pdf(self, x):
        p = self.alpha*self.gaussian1.pdf(x)
        p += (1-self.alpha)*self.gaussian2.pdf(x)
        return p
    

def conditional_correlation_sampling_2D(num=1, rho=0):
    samples = []
    rho = max(min(1,rho),-1)
    p = abs(rho)
    # print(p)
    uniform = Uniform()
    binary= Bernoulli(n=1, p=p)
    
    u1 = uniform.sample(num)[0] # u1 ~ Uniform(0,1)
    condition = binary.sample(num)[0] # c ~ Bernoulli(1,p)
    v = uniform.sample(num)[0] # v ~ Uniform(0,1)
    if abs(rho) < epsilon:
        u2 = v
    elif rho > 0: 
        u2 = condition*u1 + (1-condition)*v
    else:
        u2 = condition*(1.0-u1) + (1-condition)*v
    
    u1 = np.array(u1,dtype=float)
    u2 = np.array(u2,dtype=float)

    samples = np.vstack([u1,u2])

    return samples
    
    
def solve(func, y, bound=[-0.1,1.1], eps=1e-2):
# solve y = func(x) if func is strictly increasing
    def g(x):
        return func(x) - y
    
    def zero(x):
        return abs(x) < epsilon
    
    low = bound[0]
    high = bound[1]
    mid = 0.5*(low + high)

    g_low = g(bound[0])
    g_high = g(bound[1])
    g_mid = g(0.5*(low + high))

    if zero(g_mid):
        return mid
    elif zero(g_high):
        return high
    elif zero(g_low):
        return low
    elif g_high*g_low < 0.0:
        if g_mid*g_low < 0.0 and g_mid*g_high > 0.0:
            return solve(func, y, bound=[low,mid])
        elif g_mid*g_high < 0.0 and g_mid*g_low > 0.0:
            return solve(func, y, bound=[mid,high])
        else:
            print_warning('Not Found Solution in [%f, %f], y=%f'%(bound[0],bound[1],y))
            return 0
    else: # g_high*g_low > 0.0:
            print_warning('Not Found Solution for %f*%f > 0.0, y=%f'%(g_high,g_low,y))
            return high if abs(g_high)<abs(g_low) else low


if __name__ == '__main__':
    distribution_of_distance = BiGaussian(200,100,400,10,0.2)
    distribution_of_reflectance = Gaussian(80,20)

    x = np.linspace(0,500,1000)

    # x = np.array([0.4])

    y1 = distribution_of_distance.pdf(x)
    # print(x)

    # print(y1)
    y2 = distribution_of_reflectance.pdf(x)


    plt.plot(x, y1, color='red', linewidth=2, label='distance')
    plt.plot(x, y2, color='blue', linewidth=2, label='reflectance')
    # plt.plot(x, y3, color='green', linewidth=2, label='0.2*N(20, 14) + 0.8*N(70, 3)')

    y = np.ones(100)*0.2
    x1 = distribution_of_distance.sample(100)
    x2 = distribution_of_reflectance.sample(100)


    plt.scatter(x1,y,s=10,c='red')
    plt.scatter(x2,y,s=10,c='blue')

    JD = JointDistribution2D(distribution_x=distribution_of_distance, distribution_y=distribution_of_reflectance, rho=-0.5)
    print(JD.cov)
    N = 10
    points = JD.sample(N)
    y = np.ones(N)*0.1
    x1 = points[0,:]
    x2 = points[1,:]
    print('sampleing cov:\n', np.corrcoef(points))

    plt.scatter(x1,y,s=10,c='red', marker='x')
    plt.scatter(x2,y,s=10,c='blue', marker='x')


    # 添加标题和标签
    plt.title('Normal Distribution')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()  # 添加图例
    plt.grid(True)  # 添加网格线
    plt.savefig('./normal.png')  # 保存图像
    plt.show()  # 显示图像