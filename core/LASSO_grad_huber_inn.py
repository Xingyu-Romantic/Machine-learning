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
