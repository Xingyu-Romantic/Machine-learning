import numpy as np
import time

opt = {'itr': 20, 'itr_inn': 2500, 'sigma': 1, 
       'tol0': 1e-1, 'gamma': 1, 'verbose': 2}
def prox(x, mu):
    y = np.maximum(np.abs(x) - mu, 0)
    res = np.sign(x) * y
    return res
def BP_ALM(x0, A, b, opts):
    for i in opt.keys():
        if opts.get(i, -1) == -1:
            opts[i] = opt[i]
    sigma = opts['sigma']
    gamma = opts['gamma']
    # 迭代准备
    k = 0
    tt = time.time()
    out = {}
    # 计算并记录初始时刻都约束违反度
    out['feavec'] = np.linalg.norm(np.matmul(A, x0) -b)
    x = x0.copy()
    lambda_ = np.zeros(b.shape)
    out['itr_inn'] = 0
    
    # 记录迭代过程的优化变量x的值
    itervec = np.zeros((len(x0), opts['itr'] + 1))
    L = sigma * max(np.linalg.eig(np.matmul(A.T, A))[0])
    # 迭代主循环
    while k < opts['itr']:
        Axb = np.matmul(A, x) - b
        c = Axb + lambda_ / sigma
        g = sigma * np.matmul(A.T, c)
        tmp = 0.5 * sigma * np.linalg.norm(c, 2) ** 2
        f = np.linalg.norm(x, 1) + tmp
        
        nrmG = np.linalg.norm(x - prox(x - g, 1), 2)
        tol_t = opts['tol0'] * 10 ** (-k)
        t = 1 / L
        # 子问题求解的近似点梯度法
        Cval = tmp
        Q = 1 
        k1 = 0
        while k1 < opts['itr_inn'] and nrmG > tol_t:
            gp = g.copy()
            xp = x.copy()
            x = prox(xp - t * gp, t)
            nls =  1
            while True:
                tmp = 0.5 * sigma * np.linalg.norm(np.matmul(A , x) - b + lambda_ / sigma, 2)**2
                if tmp <= Cval + np.matmul(g.T, x - xp) + 0.5 * sigma / t * np.linalg.norm(x - xp, 2)**2 or nls == 5:
                    break
                t = 0.2 * t
                nls += 1
                x = prox(xp - t * g, t)
            f = tmp + np.linalg.norm(x, 1)
            nrmG = np.linalg.norm(x - xp, 2) / t
            Axb = np.matmul(A, x) - b
            c = Axb + lambda_ / sigma
            g = sigma * np.matmul(A.T, c)
            dx = x - xp
            dg = g - gp
            dxg = np.abs(np.matmul(dx.T, dg))
            if dxg > 0:
                if k % 2 == 0:
                    t = np.linalg.norm(dx, 2) **2 / dxg
                else:
                    t = dxg / np.linalg.norm(dg, 2) ** 2
            t = min(max(t, 1 / L), 1e12)
            Qp = Q
            Q = gamma * Qp + 1
            Cval = (gamma * Qp * Cval + tmp) / Q
            k1 += 1
            if opts['verbose'] > 1:
                print('itr_inn: %d\tfval: %e\t nrmG: %e\n'%(k1, f,nrmG))
        if opts['verbose']:
            print('itr_inn: %d\tfval: %e\t nrmG: %e\n'%(k1, f,nrmG))
        lambda_ = lambda_ + sigma * Axb
        k += 1
        out['feavec'] = np.vstack((out['feavec'], np.linalg.norm(Axb)))
        itervec[:, k] = x.flatten()
        out['itr_inn'] += k1
    out['tt'] = time.time() - tt
    out['fval'] = f
    out['itr'] = k
    out['itervec'] = itervec
    return [x, out]
    

import random
import numpy as np
import scipy.sparse
random.seed(97006855)

m = 512
n = 1024
A = np.random.randn(m, n)
u = scipy.sparse.rand(n,1,0.1).toarray()
b = np.matmul(A, u)
x0 = np.random.randn(n, 1)

optsp = {}
optsp['verbose'] = 1
optsp['gamma'] = 0.85
optsp['itr'] = 7

[x, out] = BP_ALM(x0, A, b, optsp)
k1 = out['itr']
data1 = out['feavec'][1:k1]
tmp = np.hstack((x0, out['itervec'][:, 1:k1])) - np.matmul(u, np.ones((1, k1)))
#tmp = [x0.flatten(), out['itervec'][:, 1:k1]] - np.matmul(u, np.ones((1, k1)))
tmp2 = np.sum(tmp * tmp)
data2 = np.sqrt(tmp2)