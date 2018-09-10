# -*- coding: utf-8 -*-
"""
Test LSGD with arbitrary order smoothing on the following quadratic function:
    x_1^2/1^2 + x_2^2/10^2 + ... + x_{2n-1}^2/1^2 + x_{2n}^2/10^2
Usage: python LSGD.py [-n 100] [-sigma 10] [-order 2]
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Laplacian smoothing')
ap = parser.add_argument
ap('-n', help='dim.', type=int, default=100)
ap('--maxiter', help='max. iter', type=int, default=2.5e4)
ap('-sigma', help='sigma.', type=float, default=10.0)
ap('-order', help='smoothing order', type=int, default=5)
opt = vars(parser.parse_args())

"""
Quadratic function
"""
def quad_fun(x):
    x = np.asarray(x)
    r = 0.
    for i in range(len(x)):
        coef = 1. if i%2 else 10.
        r += ((x[i]-2.)/coef)**2
    return r


def quad_dfun(x):
    der = np.zeros_like(x, dtype=np.float32)
    for i in range(len((x))):
        coef = 1. if i%2 else 10.
        eps = 0.1
        der[i] = 2*(x[i]-2.)/(coef**2) + eps*np.random.normal()
    return der


def quad(x):
    return quad_fun(x), quad_dfun(x)

def gd_smoothing(k, x, y, func, vec, order, sigma, p, cache):
    """
    Smoothing with I + (-1)^(order)*sigma* Delta^(order)
    """
    dt = p['dt']
    f, df = func(x)
    dt = dt/(10**(k/10000)) # Learning rate decay
    dt = dt*(max(1, k-1))/max(k, 1)
    if order > 0:
        # Use FFT to solve high order smoothing
        df = np.squeeze(np.real(np.fft.ifft(np.fft.fft(df)/(1+(-1)**order*sigma*np.fft.fft(vec)))))
    _x = x - dt*df
    _y = y
    return f, df, _x, _y


def runner(x0, func, optim, vec, order, sigma, p):
    p['tol'] = p.get('tol', 1e-12)
    p['dt'] = p.get('dt', 1e-2)
    
    r = dict(xs=[x0], ys=[0*x0], fs=[func(x0)[0]], dfs=[func(x0)[1]])
    fs, dfs, xs, ys = r['fs'], r['dfs'], r['xs'], r['ys']
    print('fs and dfs: ', fs, dfs)
    k = 0
    err = 1
    while fs[-1] > p['tol'] and k < p['maxiter']:
        f, df, x, y = optim(k, xs[-1], ys[-1], func, vec, order, sigma, p, cache=r)
        fs.append(f)
        dfs.append(df)
        xs.append(x)
        ys.append(y)
        err = np.linalg.norm(df)
        
        if k % 100 == 0:
            print('[%d] f: %f, df: %f'%(k, f, err))
        k += 1
    print('--------------')
    return r


n = opt['n']
ndim = n
vec = np.zeros(shape=(1, ndim))
x0 = -np.ones(n) # Starting point
order = opt['order']
if order >= 1:
    Mat = np.zeros(shape=(order, 2*order+1))
    Mat[0, order-1] = 1.; Mat[0, order] = -2.; Mat[0, order+1] = 1.

    for i in range(1, order):
        Mat[i, order-i-1] = 1.; Mat[i, order+i+1] = 1.
        Mat[i, order] = Mat[i-1, order-1] - 2*Mat[i-1, order] + Mat[i-1, order+1]
        
        Mat[i, order-i] = -2*Mat[i-1, order-i] + Mat[i-1, order-i+1]
        Mat[i, order+i] = Mat[i, order-i]
        
        for j in range(0, i-1):
            Mat[i, order-j-1] = Mat[i-1, order-j-2] - 2*Mat[i-1, order-j-1] + Mat[i-1, order-j]
            Mat[i, order+j+1] = Mat[i, order-j-1]
    
    for i in range(order+1):
        vec[0, i] = Mat[-1, order-i]
    
    
    for i in range(order):
        vec[0, -1-i] = Mat[-1, order-i-1]

sigma = opt['sigma']

print("=================== LS GD ========================")
r_gdlap1 = runner(x0, func=quad, optim=gd_smoothing,  vec=vec, order=order, sigma=sigma,
                 p=dict(dt=1e-3, maxiter=opt['maxiter'], reset=True))

ax = plt.subplot(111, xlabel='x', ylabel='y', title='title')
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(20)
plt.figure(1, figsize=(7,6))
plt.clf()
plt.plot(r_gdlap1['fs'], 'g', lw=1, label='Smoothing')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlim([10, opt['maxiter']])
plt.grid()
plt.xlabel('iterations')
plt.ylabel('|f(x) - f(x*)|')
plt.show()
