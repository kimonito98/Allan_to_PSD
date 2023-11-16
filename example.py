from algorithm import compute_mus, generate_psd, generate_clock_noise
import numpy as np

clocks = dict()

# Orolia: https://safran-navigation-timing.com/product/rafs/
clocks['RAFS'] = dict()
clocks['RAFS']['taus'] = [10, 30, 120, 480, 1.92e3, 1.54e4, 6.14e4, 1.23e5, 2.46e5, 4.92e5, 9.83e5 ]
clocks['RAFS']['adevs'] = [1e-12, 3.99e-13, 1.66e-13, 8.85e-14, 4.15e-14, 1.62e-14, 8.99e-15, 9.73e-15, 1.17e-14, 1.26e-14, 1.36e-14]
clocks['RAFS']['vartype'] = 'Overlapping Hadamard'
clocks['RAFS']['mus'] = compute_mus(clocks['RAFS']['taus'], clocks['RAFS']['adevs'])

# AccuBeat https://www.accubeat.com/uso
clocks['USO'] = dict()
clocks['USO']['taus'] = [4, 8, 16, 32, 64, 128, 256, 512, 1.02e3, 2.05e3, 4.1e3, 8.19e3, 1.64e4, 3.28e4, 6.55e4]
clocks['USO']['adevs'] = [ 1.07e-13, 1.04e-13, 9.84e-14, 1.04e-13, 1.11e-13, 1.21e-13, 1.33e-13, 1.49e-13, 1.6e-13, 2.17e-13, 3.4e-13, 6.6e-13, 1.12e-12, 1.97e-12, 3.27e-12]
clocks['USO']['vartype'] = 'Allan'
clocks['USO']['mus'] = compute_mus(clocks['USO']['taus'], clocks['USO']['adevs'])


clock = 'RAFS'
mus = clocks[clock]['mus']
taus = clocks[clock]['taus']
adevs = clocks[clock]['adevs']
vartype = clocks[clock]['vartype']

noise_semples_factor = 10

# compute psd from adev
Sy, fr_nodes, alphas, Hi, Bi = generate_psd(mus, taus[1:-1], adevs[0], vartype) 

# generate phase noise and compute adev
noise, timestep = generate_clock_noise(taus[0], taus[-1]*noise_semples_factor, fr_nodes, alphas, Hi)
times = np.arange(0, len(noise))*timestep


# Uncomment the following to plot the original ADEV and compare it to the noise-derived ADEV  
"""
import allantools
from matplotlib import pylab as plt

if vartype == 'Allan':
    (my_taus, adev_from_my_noise, adev_from_my_noise_error, ns) = allantools.adev(noise, rate = 1/timestep, data_type = 'phase', taus = taus)
elif vartype == 'Overlapping Hadamard':
    (my_taus, adev_from_my_noise, adev_from_my_noise_error, ns) = allantools.ohdev(noise, rate = 1/timestep, data_type = 'phase', taus = taus)

fig, ax = plt.subplots( figsize = (10, 6))
ax.loglog(taus, adevs, label = clock +  ' adev ')
ax.loglog(my_taus, adev_from_my_noise, label = clock + ' adev -> psd - > noise -> adev')
ax.grid(which = 'both')
ax.set_ylabel(vartype + ' [-]')
ax.set_xlabel(r'$\tau$ [s]')
ax.legend()
plt.show()
"""
