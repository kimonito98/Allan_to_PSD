import numpy as np
import scipy.integrate as integrate
from math import log, pi, sin, cos, sqrt, log10, floor, log2, pi


def compute_mus(taus, adevs):
    """ Compute the slopes between the noised of an ADEV/HDEV """
    mus = []
    for i in range(len(taus)-1):
        mus.append( 2*(log(adevs[i+1])-log(adevs[i]))/( log(taus[i+1])-log(taus[i])) )
    return mus    


def generate_psd(mus, taus, adev0, vartype):
    """ Obtain an apporoximation of the PSD from an input ADEV/HDEV based on the algorithm of
    De Marchi, F., Plumaris, M.K., Burt, E.A, Iess, L.
    A quick algorithm to compute an approximated power spectral density from an arbitrary Allan deviation
    (under review) """
    
    n_pts = 2 # number of points to discretize each interval
    n_int = len(mus)
    hi = []
    Bi = [adev0**2*taus[0]**(-mus[0])]
    for i in range(n_int-1):
        Bi.append(Bi[i]*taus[i]**(mus[i]-mus[i+1]))      
    if vartype in ('Allan', 'Overlapping Allan'):
        for i in range(n_int):
            integration_output = integrate.quad(lambda x: sin(x)**4/x**(mus[i]+3), 0, np.inf, full_output = 1)
            hi.append(Bi[i]/(2*pi**mus[i]*integration_output[0]))
    elif vartype in ('Hadamard', 'Overlapping Hadamard'):
        for i in range(n_int):
            integration_output = integrate.quad(lambda x: sin(x)**6/x**(mus[i]+3), 0, np.inf, full_output = 1)
            hi.append(Bi[i]/(16*pi**mus[i]*integration_output[0]))
    else:
        raise Exception('VarType unknown!')
    alphas = [-m-1 for m in mus]
    fr_nodes = [ ( hi[i] / hi[i+1] )**(1/(alphas[i+1]-alphas[i])) for i in range(n_int-1)]  
    alphas = list(reversed(alphas)) 
    fr_nodes = list(reversed(fr_nodes))
    fr_nodes.insert(0, 1e-30)
    fr_nodes.append(np.inf)
    hi = list(reversed(hi))
    Sy = np.zeros((n_int*n_pts))
    count = 0
    for i in range(n_int-1):
        for f in np.linspace(fr_nodes[i],fr_nodes[i+1], n_pts):
            Sy[count] = hi[i]*f**alphas[i]
            count += 1
    return Sy, fr_nodes, alphas, hi, Bi

def generate_clock_noise(tau_min, tau_max, fr_nodes, alphas, Hi):
    """ Generate phase noise from the PSD based on the algorithm from:
    Timmer, J., & Koenig, M. (1995). On generating power law noise.
    Astronomy and Astrophysics, v. 300, p. 707, 300, 707.
    """
    # number of samples (must be even, and power of 2 for ifft speed)
    n = 2**(floor(log2(tau_max/tau_min)+1))
    timestep = tau_min/2
    tau_max = (n-1)*tau_min
    f_min = 1/(n*tau_min)
    f_max = (n-1)*f_min
    n_f = n//2+1
    F = np.zeros((n), dtype=np.complex_)
    frequencies = np.linspace(f_min, f_max, n_f)
    Sxs = []
    for e, f in enumerate(frequencies):
        index = np.where( np.array(fr_nodes) >= f)[0][0]
        Sx = Hi[index-1]/(4*pi**2) *f**(alphas[index-1]-2)
        Sxs.append(Sx) 
        F[e] = sqrt(Sx/2)*(np.random.normal(0,1)+np.random.normal(0,1)*1j)
    F[0] = np.real(F[0])
    F[n_f-1] = np.real(F[n_f-1])
    F[n_f:] = np.flip(np.conjugate(F[1:n_f-1]))
    F = F/sqrt(tau_min)
    noise = np.real(np.fft.ifft(F)*sqrt(n-1))
    return noise, timestep