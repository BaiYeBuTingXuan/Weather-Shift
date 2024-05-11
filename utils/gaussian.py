from scipy.stats import norm
from scipy.stats import multivariate_normal

import os
import sys
from os.path import join, dirname

sys.path.insert(0, join(dirname(__file__), '..'))
import numpy as np
from matplotlib import pyplot as plt
from utils import print_warning
import math
import random
import itertools
import pickle
from pathlib import Path
from tqdm import tqdm

epsilon = 1e-4
infinite = 1e5
min_sigma = 0.01
min_gap = 5 

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


    def sample(self, num=1, bound=[0,200,0,1000]):
        samples = conditional_correlation_sampling_2D(num,self.rho)
    
        samples[0, :] = self.distribution_x.ppf(samples[0, :])
        samples[1, :] = self.distribution_y.ppf(samples[1, :])
        # print(samples[0, :])
        valid_indices = np.where(
            (bound[0] <= samples[0, :]) & (samples[0, :] <= bound[1]) &
            (bound[2] <= samples[1, :]) & (samples[1, :] <= bound[3]))[0]
    
        samples = samples[:, valid_indices]

        # print(samples)
        if len(samples[0, :] ) < num:
            samples_again = self.sample(num-len(samples[0, :]), bound)
            samples = np.concatenate([samples_again,samples], axis=1)
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
    
    def sample(self, num=1, dead_zone = 0.10):
        # TODO: relate x to mu and sigma
        s = [random.random()*(1-2*dead_zone) + dead_zone for _ in range(num)]
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
        
    def pdf(self, x):
        p = self.p
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
    
    def sample(self, num=1):
        uniform = Uniform()
        s = self.ppf(uniform.sample(num))
        return s

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

        self.var1 = self.sigma1 * self.sigma1
        self.var2 = self.sigma2 * self.sigma2


        self.gaussian1 = Gaussian(mu1, sigma1)
        self.gaussian2 = Gaussian(mu2, sigma2)

        self.mu = mu1*alpha + mu2*(1-alpha)
        self.var = (self.sigma**2)*alpha + (self.sigma2**2)*(1-alpha) + 2*alpha*(1-alpha)*(mu1-mu2)*(mu1-mu2)
        self.sigma = np.sqrt(self.var)

    def cdf(self, x):
        p = self.alpha*self.gaussian1.cdf(x)
        p += (1-self.alpha)*self.gaussian2.cdf(x)
        return p
    
    def ppf(self, p):
        bound = [-50,500]
        # bound.append(min(self.mu1-100*self.sigma1, self.mu2-100*self.sigma2))
        # bound.append(max(self.mu1+100*self.sigma1, self.mu2+100*self.sigma2))
        if type(p) == float or type(p) == int:
            x = binary_solve(self.cdf, p, bound=bound)
        else:
            p = np.array(p,dtype=float)
            x = self.numpy_quick_solve(p)
        # else:
        #     x = []
        #     # print(p)
        #     p = np.array(p)
        #     p = p.tolist()

        #     for pi in p:
        #         x.append(binary_solve(self.cdf, pi,bound=[0,200]))
            # print(x)
        return x


    
    def pdf(self, x):
        p = self.alpha*self.gaussian1.pdf(x)
        p += (1-self.alpha)*self.gaussian2.pdf(x)
        return p
    
    def sample(self, num=1):
        uniform = Uniform()
        # print(uniform.sample(num))
        s = self.ppf(uniform.sample(num))
        
        return s

        # b = Bernoulli(n=1, p=self.alpha)
        # conditions = np.array(b.sample(num))
        # # print(conditions)
        
        # x = np.random.normal(self.mu1, self.sigma1, size=num)*conditions 
        # y = np.random.normal(self.mu2, self.sigma2, size=num)*(1-conditions)
        # z = x + y
        # return z
    def numpy_quick_solve(self,y):
        if abs(self.mu1-self.mu2)<0.5*self.sigma1+0.5*self.sigma2:
            apporximate = Gaussian(mu=self.mu, sigma=self.sigma)
            return apporximate.ppf(y)
        # elif abs(self.mu1-self.mu2)<1.5*self.sigma1+1.5*self.sigma2:
        #     r = []
        #     for yi in y:
        #         r.append(binary_solve(self.cdf, yi, bound=[0,128]))
        #     return np.array(r,dtype=float)
        else:
            if self.mu1 < self.mu2:
                alpha = self.alpha
                f1 = self.gaussian1.ppf
                f2 = self.gaussian2.ppf
            else:
                alpha = 1-self.alpha
                f1 = self.gaussian2.ppf
                f2 = self.gaussian1.ppf
            
            c = np.where(y < alpha, 1, 0)

            r1 = f1(y/alpha)
            r1[np.isnan(r1)] = 0

            r2 = f2((y - alpha)/(1-alpha))
            r2[np.isnan(r2)] = 0

            r = c * r1 + (1-c) * r2

            # print(r)
        return r

    def quick_solve(self, y):
        if abs(self.mu1-self.mu2)<3*self.sigma1+3*self.sigma2:
            # print_warning('larger error in quick solve')
            pass
        if abs(self.mu1-self.mu2)<0.5*self.sigma1+0.5*self.sigma2:
            apporximate = Gaussian(mu=self.mu, sigma=self.sigma)
            return apporximate.ppf(y)
        # elif abs(self.mu1-self.mu2)<1*self.sigma1+1*self.sigma2:
        #     return binary_solve(self.cdf, y, bound=[0,200])
        else:
            if self.mu1 < self.mu2:
                alpha = self.alpha
                if y < alpha:
                    return self.gaussian1.ppf(y/alpha)
                else:
                    return self.gaussian2.ppf((y - alpha)/(1-alpha))
            else:
                alpha = 1-self.alpha
                if y < alpha:
                    return self.gaussian2.ppf(y/alpha)
                else:
                    return self.gaussian1.ppf((y - alpha)/(1-alpha))
            

class JointBiGaussianGaussian(JointDistribution2D):
    def __init__(self, distribution_x: BiGaussian, distribution_y: Gaussian, rho) -> None:
        super(JointBiGaussianGaussian,self).__init__(distribution_x, distribution_y, rho)
        self.alpha = self.distribution_x.alpha

        # from math import sqrt
        # A = self.alpha*self.distribution_x.mu1*self.distribution_y.mu + (1-self.alpha)*self.distribution_x.mu2*self.distribution_y.mu
        # B = self.alpha*self.distribution_x.sigma1*self.distribution_y.sigma + (1-self.alpha)*self.distribution_x.sigma2*self.distribution_y.sigma
        # C = self.alpha*(1-self.alpha)*(self.distribution_x.mu1**2+self.distribution_x.mu2**2)
        # C += 2*self.alpha*(1-self.alpha)*self.distribution_x.mu1*self.distribution_x.mu2
        # C += 2*self.alpha*(self.distribution_x.sigma1**2) + 2*(1-self.alpha)*(self.distribution_x.sigma2**2)
        # C *= self.distribution_y.sigma**2

        # rho_ = (sqrt(C)*rho-A)/B
        # print(rho,rho_)
        # print(C)
        
        self.mu1 = [self.distribution_x.mu1,self.distribution_y.mu]
        self.cov1 = np.zeros([2,2], dtype=float)
        self.cov1[0][0] = self.distribution_x.sigma1*self.distribution_x.sigma1
        self.cov1[0][1] = self.distribution_x.sigma1*self.distribution_y.sigma*rho
        self.cov1[1][0] = self.distribution_x.sigma1*self.distribution_y.sigma*rho
        self.cov1[1][1] = self.distribution_y.sigma*self.distribution_y.sigma
        # self.cov1 = self.cov1/self.alpha


        self.mu2 = [self.distribution_x.mu2,self.distribution_y.mu]
        self.cov2 = np.zeros([2,2], dtype=float)
        self.cov2[0][0] = self.distribution_x.sigma2*self.distribution_x.sigma2
        self.cov2[0][1] = self.distribution_x.sigma2*self.distribution_y.sigma*rho
        self.cov2[1][0] = self.distribution_x.sigma2*self.distribution_y.sigma*rho
        self.cov2[1][1] = self.distribution_y.sigma*self.distribution_y.sigma
        # self.cov2 = self.cov2 / (1-self.alpha)

        self.cov = self.alpha*self.cov1 + (1-self.alpha)*self.cov2
        
    
    def sample(self, num=1, bound=[0, 100, 0, 1000]):
        # uniform = Uniform() 
        # samples = [uniform.sample(num),uniform.sample(num)]
        b = Bernoulli(n=1, p=self.alpha)
        conditions = np.array(b.sample(num)).reshape(-1,1)
        
        x = np.random.multivariate_normal(self.mu1, self.cov1, size=num)*conditions 
        y = np.random.multivariate_normal(self.mu2, self.cov2, size=num)*(1-conditions)
        z = x + y

        return z.transpose()



def conditional_correlation_sampling_2D(num=1, rho=0):
    samples = []
    rho = max(min(1,rho),-1)
    p = abs(rho)
    # print(p)
    uniform = Uniform()
    binary= Bernoulli(n=1, p=p)
    
    u1 = uniform.sample(num) # u1 ~ Uniform(0,1)
    # print(u1)
    condition = binary.sample(num) # c ~ Bernoulli(1,p)
    v = uniform.sample(num) # v ~ Uniform(0,1)

    u1 = np.array(u1,dtype=float)
    u2 = np.array(v,dtype=float)
    condition = np.array(condition,dtype=float)

    if abs(rho) < epsilon:
        u2 = v
    elif rho > 0: 
        u2 = condition*u1 + (1-condition)*v
    else:
        u2 = condition*(1.0-u1) + (1-condition)*v
    # u1 = np.array(u1,dtype=float)
    # u2 = np.array(u2,dtype=float)

    samples = np.vstack([u1,u2])

    return samples

    
def binary_solve(func, y, bound=[-0.1,1.1]):
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

    if bound[1]-bound[0] < min_gap:
        return mid
    if zero(g_mid):
        return mid
    elif zero(g_high):
        return high
    elif zero(g_low):
        return low
    elif g_high*g_low < 0.0:
        if g_mid*g_low < 0.0 and g_mid*g_high > 0.0:
            return binary_solve(func, y, bound=[low,mid])
        elif g_mid*g_high < 0.0 and g_mid*g_low > 0.0:
            return binary_solve(func, y, bound=[mid,high])
        else:
            print_warning('Not Found Solution in [%f, %f], y=%f'%(bound[0],bound[1],y))
            return 0
    else: # g_high*g_low > 0.0:
            # print_warning('Not Found Solution for %f*%f > 0.0, y=%f'%(g_high,g_low,y))
            return high if abs(g_high)<abs(g_low) else low
    

def fixed_point_solve(func, y, max_df=1, x=8.0, iter=1000, eps=0.1):
# solve y = func(x) if func is strictly increasing
    def g(z):
        return z-(func(z)-y)/max_df
    
    def zero(z):
        return abs(func(z)-y) < eps
    
    print(max_df)
    print('y ',y)
    for _ in range(iter):
        if zero(x):
            print('success!')
            return x
        print(f'x ({_})',x)
        print(f'e ({_})',abs(func(x)-y))
        x = g(x)

    print(x)
    print(func(x)-y)
    print_warning(f'NO Found solution(eps={eps}) after {iter} iterations')

    return x

SAVE_PATH = Path('./gaussian_ppf').joinpath("ppf.pkl")
ALPHA = {
    'START': 0.0,
    'STOP': 1.0,
    'NUM': 10

}
MU1 = {
    'START': 0.0,
    'STOP': 60.0,
    'NUM': 60

}
MU2 = {
    'START': 0.0,
    'STOP': 60.0,
    'NUM': 60

}
SIGMA1 = {
    'START': 0.0,
    'STOP': 10.0,
    'NUM': 10
}
SIGMA2 = {
    'START': 0.0,
    'STOP': 10.0,
    'NUM': 10
}
# def search_dict_solve(alpha,mu1,mu2,sigma1,sigma2,y):
#     with open(SAVE_PATH, 'rb') as f:
#         ppf = pickle.load(f)
#     params = []
    
#     dict_list = [ALPHA,MU1,MU2,SIGMA1,SIGMA2]
#     para_list = [alpha,mu1,mu2,sigma1,sigma2]

#     vlow = []
#     vhigh = []
#     for v,d in zip(para_list,dict_list):
#         if v >= d['STOP']:
#             vlow.append(d['STOP'])
#             vhigh.append(d['STOP'])
#         else:
#             pixel = (d['STOP']-d['START'])/d['NUM']
#             index = (v-d['START'])/(pixel)
#             index_1 = int(index)
#             index_2 = int(index)+1
#             vlow.append(d['START']+index_1*pixel)
#             vhigh.append(d['START']+index_2*pixel)


#     low_params = tuple(vlow)
#     high_params = tuple(vhigh)
#     low_y = min(ppf[low_params ].keys(), key=lambda x: abs(y-x))
#     high_y = min(ppf[low_params ].keys(), key=lambda x: abs(y-x))


#     return ppf[closest_params][closest_y]


def generate_result_dict():
   
    NUM = ALPHA['NUM']*MU1['NUM']*MU2['NUM']*SIGMA1['NUM']*SIGMA2['NUM']
    alphas = np.linspace(ALPHA['START'],ALPHA['STOP'],ALPHA['NUM'])
    mu1s = np.linspace(MU1['START'],MU1['STOP'],MU1['NUM'])
    mu2s = np.linspace(MU2['START'],MU2['STOP'],MU2['NUM'])
    sigma1s = np.linspace(SIGMA1['START'],SIGMA1['STOP'],SIGMA1['NUM'])
    sigma2s = np.linspace(SIGMA1['START'],SIGMA2['STOP'],SIGMA2['NUM'])
    xs = np.linspace(0,100,300)

    ppf = {}
    bar = enumerate(itertools.product(alphas,mu1s,mu2s,sigma1s,sigma2s))
    bar = tqdm(bar, total=NUM)
    for _, (alpha,mu1,mu2,sigma1,sigma2) in bar:
        d = BiGaussian(mu1,sigma1,mu2,sigma2,alpha)
        ppf[(alpha,mu1,mu2,sigma1,sigma2)] = {d.cdf(x):x for x in xs}

    with open(str(SAVE_PATH) , 'wb') as f:
        pickle.dump(ppf,f)

def draw_example():
    distribution_of_distance = BiGaussian(2,0.5,17,2.1,0.3)
    distribution_of_reflectance = Gaussian(7,2)

    x = np.linspace(0,30,1000)

    # x = np.array([0.4])

    y1 = distribution_of_distance.pdf(x)

    y2 = distribution_of_reflectance.pdf(x)


    plt.plot(x, y1, color='red', linewidth=2, label='distance')
    plt.plot(x, y2, color='blue', linewidth=2, label='reflectance')
    # plt.plot(x, y3, color='green', linewidth=2, label='0.2*N(20, 14) + 0.8*N(70, 3)')

    y = np.ones(100)*0.2
    x1 = distribution_of_distance.sample(100)
    x2 = distribution_of_reflectance.sample(100)


    plt.scatter(x1,y,s=1,c='red')
    plt.scatter(x2,y,s=1,c='blue')

    JD = JointDistribution2D(distribution_x=distribution_of_distance,distribution_y=distribution_of_reflectance,rho=0)
    # JD = JointBiGaussianGaussian(distribution_x=distribution_of_distance,distribution_y=distribution_of_reflectance,rho=0.7)

    print(JD.cov)
    N = 100
    points = JD.sample(N,bound=[0,128,0,200])
    
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


if __name__ == '__main__':
    draw_example()