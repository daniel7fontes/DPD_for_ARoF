# -*- coding: utf-8 -*-
"""
======================================================================
Funções para aplicação, treinamento e verificação de desempenho da MP-DPD
======================================================================
"""

import numpy as np

from numba           import njit
from tqdm.notebook   import tqdm
from scipy.constants import pi



def calcMSE(x, y):
    """
    Estimativa do Erro Médio Quadrático entre os sinais de entrada x e de saída y
    
    Parameters
    ----------
    x : np.array
        Sinal de entrada do sistema
    y : np.array
        Sinal de saída do sistema
        
    Returns
    -------
    MSE : float
          Erro médio quadrático entre x e y [dB]
    """
    
    MSE = np.mean(np.abs(y - x)**2)
    return 10*np.log10(MSE)



def calcNMSE(x, y):
    """
    Estimativa do Erro Médio Quadrático Normalizado entre os sinais de entrada x e de saída y
    
    Parameters
    ----------
    x : np.array
        Sinal de entrada do sistema
    y : np.array
        Sinal de saída do sistema
        
    Returns
    -------
    NMSE : float
           Erro médio quadrático normalizado entre x e y [dB]
    """
    
    NMSE = np.mean(np.abs(y - x)**2) / np.mean(np.abs(x)**2)
    return 10*np.log10(NMSE)



def calcSNR_per_carrier(symbTx, symbRx, Ns):
    rx = np.reshape(symbRx, (-1, Ns))
    tx = np.reshape(symbTx.copy(), (-1, Ns))
    
    SNR_per_carrier = np.zeros(Ns)
    
    for k in range(Ns):
        SNR_per_carrier[k] = 10*np.log10(np.mean(np.abs((tx[:, k]))**2) / np.mean(np.abs((rx[:, k] - tx[:, k]))**2))

    return SNR_per_carrier



def powerAmplifier(x, g=16, σ=1.1, c=1.9, α=-345, β=0.17, q=4):

    abs_x = np.abs(x)
    phi_x = np.angle(x)

    abs_y = g * abs_x / (1 + np.abs(g * abs_x / c)**(2 * σ) ) ** (1 / (2 * σ))
    phi_y = α * abs_x**q / (1 + (abs_x / β) ** q) * (pi / 180)

    return abs_y * np.exp(1j * (phi_x + phi_y))
    


@njit
def MP_filter(x, coeff):
    L = x.size
    P, M = coeff.shape

    ind = np.arange(0, M)
    
    w = coeff.ravel()
    y = np.zeros(L, dtype=np.complex128)

    x_window = np.zeros(2 * (M - 1) + L, dtype=np.complex128)
    for i in range(L):
        x_window[i] = x[i]
        
    xk = np.zeros(P * M, dtype=np.complex128)

    for i in range(L):
        X = x_window[i - ind]
        j = 0
        for p in range(P):
            for m in range(M):
                xk[j] = X[m] * (np.abs(X[m]) ** p)
                j += 1
  
        y[i] = np.dot(xk, w)

    return y



@njit
def MP_sliding_window(x, i, P, M):
    ind = np.arange(0, M)
    N = x.size
    
    x_extend = np.zeros(2 * (M - 1) + N, dtype=np.complex128)
    for n in range(N):
        x_extend[n] = x[n]
    
    x_win = np.zeros(P * M, dtype=np.complex128)
    X = x_extend[i - ind]
    j = 0
    
    for p in range(P):
        for m in range(M):
            x_win[j] = X[m] * (np.abs(X[m]) ** p)
            j += 1

    return x_win



def LS_solver(x, y, P, M):
    """
    Cálculo dos coeficientes do polinômio de memória que identifica um sistema com entrada x e saída y aplicando-se o método dos mínimos quadrados
    
    Parameters
    ----------
    x : np.array
        Sinal de entrada do sistema
    y : np.array
        Sinal de saída do sistema
    M : int
        Parâmetro de memória do polinômio de memória
    P : int
        Parâmetro de não-linearidade do polinômio de memória
    
    Returns
    -------
    a_LS : np.array
           Matriz de coeficientes PxM do polinômio de memória que representa o sistema
    """

    N = x.size
    X = np.zeros((N, P*M), dtype = complex)
    
    for n in range(N):
      X[n,:] = MP_sliding_window(x, n, P, M)
    
    a_LS = np.linalg.inv(np.conj(np.transpose(X)) @ X ) @ np.conj(np.transpose(X)) @ y.reshape((N, 1))
    a_LS = a_LS.reshape((P, M))
    
    return a_LS



@njit 
def get_psi_vec(u, x, a_kl, i, P, M):
    """
    Retorna o vetor de Psi[k] para o cálculo dos coeficientes da DPD na configuração DLA
    
    Parameters
    ----------
    u    : np.array
           Sinal de treinamento
    x    : np.array
           Sinal na saída do DPD
    a_pm : np.array 
           Coeficientes do PA identificado
    k    : int
           Instante do cálculo de Psi[k]
    M    : int
           Parâmetro de memória do polinômio de memória
    P    : int
           Parâmetro de não-linearidade do polinômio de memória
    
    Returns
    -------
    psi : np.array
          Vetor psi(i)
    """
    
    K, L = a_kl.shape
    psi = np.zeros(P*M, dtype=np.complex128)

    ind = np.arange(0, L)
    N = x.size
    
    x_extend = np.zeros(2 * (L - 1) + N, dtype=np.complex128)
    for n in range(N):
        x_extend[n] = x[n]
    
    
    for l in range(L):
        x_win = x_extend[i - ind][l]

        for k in range(K):
            psi += (1 + k/2) * a_kl[k, l] * np.abs(x_win)**k * MP_sliding_window(u, i - l, P, M)
    
    return psi



def MP_training(u, param, y = None):
    M           = param.M
    P           = param.P
    
    N           = param.N
    numIter     = param.numIter
    
    mu          = param.mu
    lbd         = param.lbd
    S           = param.S
    
    alg         = param.alg
    a_kl        = param.a_kl
    directLearn = param.directLearn
    
    pgrsBar     = param.pgrsBar
    storeCoeff  = param.storeCoeff
    
    u = u[0:N]
    y = y[0:N] if not(directLearn) else None
    
    #w = np.random.rand(M*P, 1).astype(complex)
    w = np.zeros((P*M, 1), dtype = complex)
    w[0] = 1
    
    errSq_hist = np.zeros(numIter)
    w_hist     = np.zeros((numIter, P*M), dtype = complex)
    
    for i in tqdm(range(numIter), disable = not(pgrsBar) ):
        w, errSq = coreMP_training(u, 
                                   y, 
                                   w, 
                                   N, 
                                   M, 
                                   P, 
                                   mu, 
                                   lbd, 
                                   S, 
                                   a_kl,
                                   alg, 
                                   directLearn)
        
        w_hist[i,:] = w.ravel().copy()
        errSq_hist[i] = np.mean(errSq)
        #print(f"Iter {i+1} - MSE = {10*np.log10(np.nanmean(errSq)):.3f} dB")
    
    
    return (w, errSq, w_hist, errSq_hist) if storeCoeff else (w, errSq)
    


def coreMP_training(u, y, w, N, M, P, mu, lbd, S, a_kl, alg, directLearn):
    
    errSq = np.zeros(N)
    K, L = np.shape(a_kl)
    
    # Intermediate arrays
    x = np.zeros(N, dtype = complex)
    
    if directLearn:
        for i in range(N):
            u_win = MP_sliding_window(u, i, P, M)
            x[i] = np.dot(u_win, np.conj(w))[0]
                        
            # Modelo de canal RoF
            x_win = MP_sliding_window(x, i, K, L)
            z = np.dot(x_win, a_kl.reshape((K*L, 1)))
            
            err = u[i] - z
    
            # Calc of psi vector
            psi = get_psi_vec(u, x, a_kl, i, P, M)

            # Algoritmo adaptativo
            if alg == "NFxRLS":
                g = (1/lbd) * (S @ psi.reshape(P*M, 1) ) / ( 1 + (1/lbd) * np.conj(psi.reshape(1, P*M)) @ S @ psi.reshape(P*M, 1) )
                S = (1/lbd) * S - (1/lbd)*g.reshape(P*M, 1) @ np.conj(psi.reshape(1, P*M)) @ S
                w += g * np.conj(err)
    
            elif alg == "NFxLMS":
                w += mu * np.conj(err) * psi.reshape((P*M, 1))

            errSq[i] = np.abs(err)**2
    
    else:
        for i in range(N):
            y_win = MP_sliding_window(y, i, P, M)
            z   = np.dot(y_win, np.conj(w))
            err = u[i] - z
            
            if alg == "RLS":
                g = (1/lbd) * (S @ y_win.reshape(P*M, 1) ) / ( 1 + (1/lbd)* np.conj(y_win.reshape(1, P*M)) @ S @ y_win.reshape(P*M, 1) )
                S = (1/lbd) * S - (1/lbd)*g.reshape(P*M, 1) @ np.conj(y_win.reshape(1, P*M)) @ S
                w += g * np.conj(err)
                
            else:
                w += mu * np.conj(err) * y_win.reshape((P*M, 1))
                
            errSq[i] = np.abs(err)**2
            
    return w, errSq