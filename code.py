import numpy as np
from scipy.integrate import quad, IntegrationWarning
from numpy.fft import ifft
import allantools
from matplotlib import pylab as plt
import warnings


def compute_psd_from_adev(adevs, taus, vartype='ADEV'):
    """
    Generate an approximation of the Power Spectral Density (PSD) from an input Allan Deviation (ADEV) or Hadamard Deviation (HDEV).
    F. De Marchi, M. K. Plumaris, E. A. Burt and L. Iess, "An Algorithm to Estimate the Power Spectral Density From Allan Deviation," 
    in IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control, vol. 71, no. 4, pp. 506-515, April 2024,  doi: 10.1109/TUFFC.2024.3372395
    https://ieeexplore.ieee.org/document/10466605
    
    Parameters
    ----------
    adevs: list of floats
        Allan Deviation (ADEV) values, representing the deviation at each tau.
    taus: list of floats
        Corresponding integration times for each ADEV value.
    vartype: str
        Type of variance, either 'ADEV' for Allan Deviation or 'HDEV' for Hadamard Deviation. Default is 'ADEV'.

    Returns
    -------
    A dictionary with the following:
        'frequency_nodes': list of floats
             Frequency nodes corresponding to the taus
        'values': list of floats
            Computed PSD values at each frequency node.
        'hi': list of floats
            PSD coefficients for each frequency interval.
        'alphas': list of floats
            Slopes in the frequency domain for each interval.

    """
    
    q = lambda z: 1 if vartype == 'ADEV' else (4/3) * (np.sin(z)**2)
    
    # Filter consecutive slopes (mus) and calculate Bi coefficients
    mus, filtered_adevs, filtered_taus = [], [adevs[0]], [taus[0]]
    for i in range(1, len(adevs)):
        mu = 2 * (np.log(adevs[i]) - np.log(adevs[i - 1])) / (np.log(taus[i]) - np.log(taus[i - 1]))
        if not mus or abs(mu - mus[-1]) > 1e-5:
            mus.append(mu)
            filtered_adevs.append(adevs[i])
            filtered_taus.append(taus[i])
        if vartype == 'ADEV' and not (-2 <= mu <= 2) or vartype == 'HDEV' and not (-2 <= mu <= 4):
            raise ValueError(f"Integral not convergent between nodes {filtered_taus[i - 1]} [s] and {filtered_taus[i]} [s]")

    Bi = [filtered_adevs[i]**2 * filtered_taus[i]**(-mus[i - 1]) for i in range(1, len(mus) + 1)]
    hi, integral_values = [], []
    for i, B in enumerate(Bi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=IntegrationWarning)
            integral_val, _ = quad(lambda z: q(z) * np.sin(z)**4 / (z**(3 + mus[i])), 0, np.inf)
        integral_values.append(integral_val)
        hi.append(B / (2 * integral_val * (np.pi**mus[i])))
    
    alphas = [-mu - 1 for mu in mus[::-1]]
    frequency_nodes = [1 / (10 * np.pi * filtered_taus[i]) * (integral_values[i] / integral_values[i + 1])**(1 / (mus[i + 1] - mus[i])) for i in range(len(mus) - 1)]
    frequency_nodes.reverse()
    psd_values = [hi[i] * frequency_nodes[i]**(alphas[i]) for i in range(len(frequency_nodes))] if len(filtered_adevs) > 2 else [hi[0] * (1 / tau)**alphas[0] for tau in filtered_taus[::-1]]
    
    return {'frequency_nodes': frequency_nodes, 'values': psd_values, 'hi': hi[::-1], 'alphas': alphas}

def psd_to_adev(hi, alphas, frequency_nodes, taus):
    """
    Convert PSD to Allan deviation values using direct numerical integration of their fundamental relationship (see NIST Special Publication 1065 eq 65)
    
    Parameters
    ----------
    hi: list of floats
        PSD coefficients for each frequency interval.
    alphas: list of floats
        Slopes in the frequency domain for each interval.
    frequency_nodes: list of floats
        Frequency nodes denoting the PSD intervals
    taus: list of floats
        Integration time values for desired PSD output

    Returns
    ----------
    A dictionary containing the tau as keys and corresponding ADEV as values.
            
            """
    
    def integrand(z, alpha):
        return (np.sin(z)**4) / (z**(2 - alpha))
    
    adev_dict = {}
    integration_frequencies = [0] + frequency_nodes + [np.inf]
    for tau in taus:
        sigma_y2 = 0
        for i, (h, alpha) in enumerate(zip(hi, alphas)):
            factor = h / (np.pi * tau)**(alpha + 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=IntegrationWarning)
                integral_value, _ = quad(integrand, np.pi * tau * integration_frequencies[i], np.pi * tau * integration_frequencies[i+1], args=(alpha,))
            sigma_y2 += factor * integral_value
        adev = np.sqrt(2 * sigma_y2)
        adev_dict[tau] = adev
    
    return adev_dict


def generate_noise_from_psd(frequency_nodes, hi, alphas, duration, timestep, output='phase'):
    """
    Generate time-domain noise from the given PSD values.
    Timmer, Jens, and Michel Koenig. "On generating power law noise." Astronomy and Astrophysics, v. 300, p. 707 300 (1995): 707.

    Parameters:
        'frequency_nodes' (list of floats): Frequency nodes corresponding to the PSD.
        'hi' (list of floats): PSD coefficients for each frequency interval.
        'alphas' (list of floats): Slopes in the frequency domain for each interval.
        'duration' (float): Duration of the generated time series in seconds.
        'timestep' (float): Timestep between samples in the generated time series in seconds (sampling rate).

    Returns:
        np.ndarray: Generated noise in the time domain.
        
    """
    
    # extrapolate the behaviour at infinity
    hi = psd_data['hi'] + [psd_data['hi'][-1]]
    alphas = psd_data['alphas'] + [psd_data['alphas'][-1]]
    
    n = int(duration / timestep)
    f1 = 1 / ((n - 1) * timestep)
    fn = 1 / (2 * timestep)
    frequencies = np.linspace(f1, fn, n // 2 + 1)
    
    S_y = np.zeros_like(frequencies)
    for i, f in enumerate(frequencies):
        # Determine which interval the frequency f belongs to
        if f < frequency_nodes[0]:
            S_y[i] = hi[0] * f ** alphas[0]
        elif f > frequency_nodes[-1]:
            S_y[i] = hi[-1] * f ** alphas[-1]
        else:
            # Find the interval that contains f
            for j in range(1, len(frequency_nodes)):
                if frequency_nodes[j - 1] <= f < frequency_nodes[j]:
                    S_y[i] = hi[j] * f ** alphas[j]
                    break    
    
    Sx = S_y / (2 * np.pi * frequencies) ** 2
    Sx[0] = 0
    
    X = np.zeros(n, dtype=complex)
    for j in range(1, n // 2 + 1):
        X[j] = np.sqrt(Sx[j]) / 2 * (np.random.normal(0, 1) + 1j * np.random.normal(0, 1))
    
    X[n // 2 + 1:] = np.conj(X[1:n // 2][::-1])
    x = ifft(X) * np.sqrt((n - 1) / timestep)
    return x.real if output == 'phase' else np.concatenate(([0], np.diff(x.real) / timestep))

# Example Usage (Accubeat USO)
taus = [4, 8, 16, 32, 64, 128, 256, 512, 1.02e3, 2.05e3, 4.1e3, 8.19e3, 1.64e4, 3.28e4, 6.55e4]
adevs = [1.07e-13, 1.04e-13, 9.84e-14, 1.04e-13, 1.11e-13, 1.21e-13, 1.33e-13, 1.49e-13, 1.6e-13, 2.17e-13, 3.4e-13, 6.6e-13, 1.12e-12, 1.97e-12, 3.27e-12]
vartype = 'ADEV'

# Example Usage (Orolia RAFS)
#taus =[10, 30, 120, 480, 1.92e3, 1.54e4, 6.14e4, 1.23e5, 2.46e5, 4.92e5, 9.83e5 ]
#adevs = [1e-12, 3.99e-13, 1.66e-13, 8.85e-14, 4.15e-14, 1.62e-14, 8.99e-15, 9.73e-15, 1.17e-14, 1.26e-14, 1.36e-14]

# calculate the "approximate" PSD from the ADEV
psd_data = compute_psd_from_adev(adevs, taus, vartype=vartype)

# recompute the ADEV from direct numerical integration of the PSD
adev_dict = psd_to_adev(psd_data['hi'], psd_data['alphas'], psd_data['frequency_nodes'], taus)

# Generate noise with the desired PSD, using
duration = taus[-1] * 100 # duration to capture full ADEV behaviour
timestep = taus[0] / 2 # Nyquist sample rate
noise = generate_noise_from_psd(psd_data['frequency_nodes'], psd_data['hi'],psd_data['alphas'], duration, timestep)
times = np.arange(0, len(noise)) * timestep

# Calculate ADEV from the generated noise
my_taus, adev_from_noise, _, _ = allantools.adev(noise, rate=1/timestep, data_type='phase', taus=taus)

# Plot results
plt.figure(figsize=(10, 6))
plt.loglog(taus, adevs, label='Original ADEV')
plt.loglog(list(adev_dict.keys()), list(adev_dict.values()), label='ADEV -> PSD -> ADEV')
plt.loglog(my_taus, adev_from_noise, label='ADEV -> PSD -> noise -> ADEV')
plt.grid(which='both')
plt.ylabel('ADEV [-]')
plt.xlabel(r'$\tau$ [s]')
plt.legend()
plt.show()
