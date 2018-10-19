import numpy as np
import matplotlib.pyplot as plt
from scipy import loadtxt

chan_na, na_spectrum = loadtxt('Na-22 Absorber5.tsv', unpack=True, skiprows=25)
chan_na = chan_na[50:]
na_spectrum = na_spectrum[50:]
fig = plt.figure()
na_spectrumerr = np.sqrt(na_spectrum)

ax = fig.add_subplot(111)

ax.set_xlabel("Channel (channel number)")
ax.set_ylabel("Counts (number of incident photons)")
ax.errorbar(chan_na,na_spectrum,na_spectrumerr,fmt='ko')
plt.title("Figure 1: Full Spectrum of A Na-22 Source \n"\
    "Attenuated with 1.6 cm of Aluminum")
plt.savefig("Na-22 Full Spectrum.pdf")
plt.show()