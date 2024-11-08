import numpy as np
from scipy.integrate import quad
from numpy.fft import ifft
import allantools
from matplotlib import pylab as plt


def compute_psd_from_adev(adevs, taus, vartype='ADEV'):
    """
    Generate an approximation of the Power Spectral Density (PSD) from an input Allan Deviation (ADEV) or Hadamard Deviation (HDEV).
    F. De Marchi, M. K. Plumaris, E. A. Burt and L. Iess, "An Algorithm to Estimate the Power Spectral Density From Allan Deviation," in IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control, vol. 71, no. 4, pp. 506-515, April 2024, doi: 10.1109/TUFFC.2024.3372395.


    Parameters:
        adevs (list): Allan Deviation (ADEV) values.
        taus (list): Corresponding tau values for the ADEV.
        vartype (str): Type of variance ('ADEV' or 'HDEV').

    Returns:
        dict: A dictionary containing:
            - 'frequencies': Corresponding frequency nodes for the PSD.
            - 'psd_values': Power Spectral Density (PSD) values.
    """
    def q_avar(z):
        return 1

    def q_hvar(z):
        return (4/3) * (np.sin(z)**2)

    if vartype == 'ADEV':
        q = q_avar
    elif vartype == 'HDEV':
        q = q_hvar
    else:
        raise ValueError("Unknown vartype. Please use 'ADEV' or 'HDEV'.")

    # Step 1: Calculate slopes (mu_i)
    mus = [2 * (np.log(adevs[i]) - np.log(adevs[i - 1])) / (np.log(taus[i]) - np.log(taus[i - 1])) for i in range(1, len(adevs))]
    
    # Step 2: Calculate Bi coefficients
    Bi = [adevs[i]**2 * taus[i]**(-mus[i - 1]) for i in range(1, len(mus) + 1)]
        
    # Step 3: Calculate hi coefficients in the frequency domain
    hi = []
    integral_values = []
    for i in range(len(Bi)):
        integral_value, _ = quad(lambda z: q(z) * np.sin(z)**4 / (z**(3 + mus[i])), 0, np.inf)
        integral_values.append(integral_value)
        hi.append(Bi[i] / (2 * integral_value * (np.pi**mus[i])))

    # Reverse the hi values for frequency ordering
    hi.reverse()
    
    # Step 4: Calculate slopes in the frequency domain (alpha_i)
    alphas = [-mus[-(i + 1)] - 1 for i in range(len(mus))]

    # Step 5: Calculate frequency nodes (f_n-i)
    frequency_nodes = [1 / (10 * np.pi * taus[i]) * (integral_values[i] / integral_values[i + 1])**(1 / (mus[i + 1] - mus[i])) for i in range(len(mus) - 1)]
    frequency_nodes.reverse()

    # Calculate PSD values
    psd_values = [hi[i] * frequency_nodes[i]**(alphas[i]) for i in range(len(frequency_nodes))]
    
    return {'frequencies': frequency_nodes, 'psd_values': psd_values}


def generate_noise_from_psd(psd_values, frequencies, duration, timestep, output='phase'):
    """
    Generate time-domain noise from the given PSD values.
    Timmer, Jens, and Michel Koenig. "On generating power law noise." Astronomy and Astrophysics, v. 300, p. 707 300 (1995): 707.

    Parameters:
        psd_values (list): Power Spectral Density (PSD) values.
        frequencies (list): Corresponding frequencies for PSD values.
        duration (float): Duration of the time series to generate, in seconds.
        timestep (float): Timestep between samples, in seconds.
        output (str): Type of output ('phase' or 'frequency').

    Returns:
        np.ndarray: Generated noise in the time domain.
    """
    # Determine number of samples
    n = int(duration / timestep)
    
    # Generate equally spaced frequencies for FFT
    f1 = 1 / ((n - 1) * timestep)
    fn = 1 / (2 * timestep)
    freq_nodes = np.linspace(f1, fn, n // 2 + 1)
    
    # Interpolate PSD values to match the equally spaced frequencies
    log_freq_nodes = np.log10(frequencies)
    log_psd_values = np.log10(psd_values)
    log_interpolated_psd = np.interp(np.log10(freq_nodes), log_freq_nodes, log_psd_values)
    interpolated_psd = 10**log_interpolated_psd
    
    # Calculate Sx(f) from Sy(f) using the relationship Sx(f) = Sy(f) / (2 * pi * f)^2
    Sx = interpolated_psd / (2 * np.pi * freq_nodes)**2
    Sx[0] = 0  # Set DC component to zero
    
    # Generate random noise in the frequency domain with the PSD shape
    X = np.zeros(n, dtype=complex)
    for j in range(1, n // 2 + 1):
        Nj_R = np.random.normal(0, 1)
        Nj_I = np.random.normal(0, 1)
        X[j] = np.sqrt(Sx[j]) / 2 * (Nj_R + 1j * Nj_I)
    
    # Use symmetry for the inverse FFT
    X[n // 2 + 1:] = np.conj(X[1:n // 2][::-1])
    
    # Compute inverse FFT to get time-domain signal
    x = ifft(X) * np.sqrt((n - 1) / timestep)
    
    # Return either phase or frequency fluctuations
    if output == 'phase':
        return x.real
    else:
        # Frequency fluctuations from phase (differentiation in discrete time)
        y = np.diff(x.real) / timestep
        return np.concatenate(([0], y))

    
# Example Usage
clock = "USO"
taus = [4, 8, 16, 32, 64, 128, 256, 512, 1.02e3, 2.05e3, 4.1e3, 8.19e3, 1.64e4, 3.28e4, 6.55e4]
adevs = [ 1.07e-13, 1.04e-13, 9.84e-14, 1.04e-13, 1.11e-13, 1.21e-13, 1.33e-13, 1.49e-13, 1.6e-13, 2.17e-13, 3.4e-13, 6.6e-13, 1.12e-12, 1.97e-12, 3.27e-12]

#clock = "RAFS"
#taus =[10, 30, 120, 480, 1.92e3, 1.54e4, 6.14e4, 1.23e5, 2.46e5, 4.92e5, 9.83e5 ]
#adevs = [1e-12, 3.99e-13, 1.66e-13, 8.85e-14, 4.15e-14, 1.62e-14, 8.99e-15, 9.73e-15, 1.17e-14, 1.26e-14, 1.36e-14]

vartype = 'ADEV'

# Generate PSD from ADEV
psd_data = compute_psd_from_adev(adevs, taus, vartype=vartype)
print("Frequencies:", psd_data['frequencies'])
print("PSD Values:", psd_data['psd_values'])

# Generate time-domain noise from PSD
duration = taus[-1]*50  # 
timestep = taus[0]/2    # 
noise = generate_noise_from_psd(psd_data['psd_values'], psd_data['frequencies'], duration, timestep)
times = np.arange(0, len(noise))*timestep

if vartype == 'ADEV':
    (my_taus, adev_from_my_noise, adev_from_my_noise_error, ns) = allantools.adev(noise, rate = 1/timestep, data_type = 'phase', taus = taus)


fig, ax = plt.subplots( figsize = (10, 6))
ax.loglog(taus, adevs, label = clock +  ' adev ')
ax.loglog(my_taus, adev_from_my_noise, label = clock + ' adev -> psd - > noise -> adev')
ax.grid(which = 'both')
ax.set_ylabel(vartype + ' [-]')
ax.set_xlabel(r'$\tau$ [s]')
ax.legend()
plt.show()
