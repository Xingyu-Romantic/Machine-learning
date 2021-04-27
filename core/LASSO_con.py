import time
import numpy as np
def LASSO_grad_huber_inn(x, A, b, mu, mu0, opt):
    for i in opts.keys():
        if opt.get(i, -1) == -1:
            opt[i] = opts[i]
    tic = time.time()
    r = np.matmul(A, x) - b
    g = np.matmul(A.T, r)
    huber_g = np.sign(x)
    idx = abs(x) < opt['sigma']
    huber_g[idx] = x[idx] / opt['sigma']
    
    g = g + mu * huber_g
    nrmG= np.linalg.norm(g, 2)
    
    f = 0.5 * np.linalg.norm(r, 2) ** 2 + \
                mu * (np.sum(np.square(x[idx])/(2 * opt['sigma'])) \
                      + np.sum(np.abs(x[abs(x) >= opt['sigma']]) - \
                               opt['sigma'] / 2))
    out = {}
    
    out['fvec'] = 0.5 * np.linalg.norm(r, 2) ** 2 + mu0 * np.linalg.norm(x, 1)
    alpha = opt['alpha0']
    eta = 0.2
    
    rhols = 1e-6
    gamma = 0.85
    Q = 1
    Cval = f
    for k in range(opt['maxit']):
        fp = f
        gp = g
        xp = x
        nls = 1
        while 1:
            x = xp - alpha * gp
            r = np.dot(A, x) - b
            g = np.dot(A.T, r)
            huber_g = np.sign(x)
            idx = abs(x) < opt['sigma']
            huber_g[idx] = x[idx] / opt['sigma']
            f = 0.5 * np.linalg.norm(r, 2) ** 2 + \
                mu * (np.sum(x[abs(x) >= opt['sigma']] - opt['sigma'] / 2))
            g = g + mu * huber_g
            if f <= Cval - alpha * rhols * nrmG ** 2 or nls >= 10:
                break
            alpha = eta * alpha 
            nls += 1
        nrmG = np.linalg.norm(g, 2)
        forg = 0.5 * np.linalg.norm(r, 2) ** 2 + mu0 * np.linalg.norm(x, 1)
        out['fvec'] = [out['fvec'], forg]
        if opt['verbose']:
            print('%4d\t %.4e \t %.1e \t %.2e \t %2d \n'%(k, f, nrmG, alpha, nls))
        if nrmG < opt['gtol'] or abs(fp - f) < opt['ftol']:
            break
        dx = x - xp
        xg = g - gp
        dxg = abs(np.matmul(dx.T, dx))
        if dxg > 0:
            if k % 2 == 0:
                alpha = np.matmul(dx.T, dx) / dxg
            else:
                alpha = dxg / np.matmul(dg.T, dg)
            alpha = max(min(alpha, 1e12), 1e-12)
        Qp = Q
        Q = gamma * Qp + 1
        Cval = (gamma * Qp * Cval + f) / Q
    out['flag'] = k == opt['maxit']
    out['fval'] = f
    out['itr'] = k
    out['tt'] = time.time() - tic
    out['nrmG'] = nrmG
    return [x, out]

optsp = {'maxit': 30, 'maxit_inn':1, 'ftol': 1e-8, 'gtol': 1e-6, 
        'factor': 0.1, 'verbose': 1, 'mul': 100, 'opts1':{},
        'etaf': 1e-1, 'etag': 1e-1}
optsp['gtol_init_ratio'] = 1 / optsp['gtol']
optsp['ftol_init_ratio'] = 1e5
def prox(x, mu):
    y = np.max(np.abs(x) - mu, 0)
    y = np.dot(np.sign(x), y)
    return y
def Func(A, b, mu0, x):
    w = np.dot(A, x) - b
    f = 0.5 * (np.matmul(w.T, w)) + mu0 * np.linalg.norm(x, 1)
    return f
def LASSO_con(x0, A, b, mu0, opts):
    L = max(np.linalg.eig(np.matmul(A.T, A))[0])
    for i in optsp.keys():
        if opts.get(i, -1) == -1:
            opts[i] = optsp[i]
    if not opts['alpha0']: opts['alpha0'] = 1 / L
    out = {}
    out['fvec'] = []
    k = 0
    x = x0
    mu_t = opts['mul']
    tic = time.time()
    f = Func(A, b, mu_t, x)
    opts1 = opts['opts1']
    opts1['ftol'] = opts['ftol'] * opts['ftol_init_ratio']
    opts1['gtol'] = opts['gtol'] * opts['gtol_init_ratio']
    out['itr_inn'] = 0
    while k < opts['maxit']:
        opts1['maxit'] = opts['maxit_inn']
        opts1['gtol'] = max(opts1['gtol'] * opts['etag'], opts['gtol'])
        opts1['ftol'] = max(opts1['ftol'] * opts['etaf'], opts['ftol'])
        opts1['verbose'] = opts['verbose'] > 1
        opts1['alpha0'] = opts['alpha0']
        if opts['method'] == 'grad_huber':
            opts1['sigma'] = 1e-3 * mu_t
        fp = f
        [x, out1] = LASSO_grad_huber_inn(x, A, b, mu_t, mu0, opts1)
        f = out1['fvec'][-1]
        out['fvec'].extend(out1['fvec'])# = [out['fvec'], out1['fvec']]
        k += 1
        nrmG = np.linalg.norm(x - prox(x - np.matmul(A.T, (np.matmul(A, x) - b)), mu0),2)
        if opt['verbose']:
            print('itr: %d\tmu_t: %e\titr_inn: %d\tfval: %e\tnrmG: %.1e\n'%(k, mu_t, out1.itr, f, nrmG))
        if not out1['flag']:
            mu_t = max(mu_t * opts['factor'], mu0)
        if mu_t == mu0 and (nrmG < opts['gtol'] or abs(f - fp) < opts['ftol']):
            break
        out['itr_inn'] = out['itr_inn'] + out1['itr']
    out['fval'] = f
    out['tt'] = time.time() - tic
    out['itr'] = k
    return [x, out]
if __name__ == '__main__':
    print(dir())
