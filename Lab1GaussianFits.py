# PHYS 21101 Lab 1 - Gamma Ray Cross Sections
# Fitting a Gaussian for Na-22 511 KeV peak
# Jonathan Rockhill
# Written 10/17/2018

import numpy as np
import matplotlib.pyplot as plt
from scipy import loadtxt, optimize

def fitfunc(p, x):
    return (p[0]/np.sqrt(2*np.pi*p[1]**2)*np.exp(-(x-p[2])**2/(2*p[1]**2))
        +p[3]*x+p[4])
def residual(p, x, y, dy):
    return (fitfunc(p, x)-y)/dy

plotba = False

plotna = True

chan_na, na_spectrum = loadtxt('Na-22 No_Absorber.tsv', unpack=True, skiprows=25)

chan_ba, ba_spectrum = loadtxt('Ba-133 No_Absorber.tsv', unpack=True, skiprows=25)

fig = plt.figure()

ba_spectrum = ba_spectrum[150:241]
chan_ba = chan_ba[150:241]
ba_spectrumerr = np.sqrt(ba_spectrum)

na_spectrum = na_spectrum[185:241]
chan_na = chan_na[185:241]
na_spectrumerr = np.sqrt(na_spectrum)


if plotba and plotna:
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    axesdict = {'Barium':ax2,'Sodium':ax1}
else:
    ax1 = fig.add_subplot(111)
    ax2 = fig.add_subplot(111)
    axesdict = {'Barium':ax2,'Sodium':ax1}

paramdict = {'Barium':[[2.5e5,6.79,190,-1.0,160],ba_spectrum,ba_spectrumerr,chan_ba],
    'Sodium':[[1.6e5,7.64,212,-1.0,200],na_spectrum,na_spectrumerr,chan_na]}

conditionaldict = {'Barium':plotba,'Sodium':plotna}

for spectrum in paramdict:
    if conditionaldict[spectrum]:
        x = paramdict[spectrum][3]
        p01 = paramdict[spectrum][0]
        Y = paramdict[spectrum][1]
        dY = paramdict[spectrum][2]
        pf1, cov1, info1, mesg1, success1 = optimize.leastsq(residual, p01,
            args = (x, Y, dY), full_output=1)
        plotfits = True
        if cov1 is None:
            print('Fit did not converge')
            print('Success code:', success1)
            print(mesg1)
            plotfits = False
        else:
            print('Fit Converged')
            chisq1 = sum(info1['fvec']*info1['fvec'])
            dof1 = len(x)-len(pf1)
            pferr1 = [np.sqrt(cov1[i,i]) for i in range(len(pf1))]
            print('Converged with chi-squared', chisq1)
            print('Number of degrees of freedom, dof =',dof1)
            print('Reduced chi-squared:', chisq1/dof1)
            print('Inital guess values:')
            print('  p0 =', p01)
            print('Best fit values:')
            print('  pf =', pf1)
            print('Uncertainties in the best fit values:')
            print('  pferr =', pferr1)
            print()    
            X = np.linspace(x.min(), x.max(), 500)
        ax = axesdict[spectrum]
        ax.errorbar(x,Y,yerr=dY,fmt='k.',capsize=2,label='Data')
        if plotfits:
            print('X ranges:', X.min(), X.max())
            print('fit($\\mu$) = ',fitfunc(pf1,pf1[2]))
            ax.plot(X,fitfunc(pf1,X),'g-',label='fit')
            ax.legend()
            textfit = '$g(x) = \\frac{N}{\\sigma\\sqrt{2\\pi}}e^{-(x-\\mu)^2}$'\
                  '$+Ax+B$\n' \
                  '$\chi^2= %.2f$ \n' \
                  '$N = %i$ (dof) \n' \
                  '$\chi^2/N = % .2f$' \
                   % (chisq1, dof1, chisq1/dof1)
            ax.text(0.01, .99, textfit, transform=ax.transAxes, fontsize=12,
                 verticalalignment='top')
        ax.set_xlabel('Channel (Channel Number)')
        ax.set_ylabel('Counts (Number of Incident Photons)')

plt.title("Figure 2: Isolated Unattenuated Na-22 511 KeV Peak with Gaussian Fit")
plt.savefig("Na-22 511 KeV Gaussian_Final.pdf")

    

plt.show()

