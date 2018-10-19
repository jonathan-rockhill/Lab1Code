# PHYS 21101 Lab 1 - Gamma Ray Cross Sections
# Analytics and Graphs
# Jonathan Rockhill
# Written 10/16/2018

import numpy as np
import matplotlib.pyplot as plt
from scipy import loadtxt, optimize
from scipy.special import erfc
import csv

def fitfunc(p, x):
    return p[0]*np.exp(-x*p[1])+p[2]
def residual(p, x, y, dy):
    return (fitfunc(p, x)-y)/dy

#If the x = 2.0 cm run is removed
origData = False

def sdresid2cm(p, x, y, dy):
    sd = np.std(residual(p,x,y,dy))
    return np.abs(residual(p,x[6],y[6],dy[6]))/sd


if origData: 
    datafile = 'Lab1Data.csv' 
    filename = 'Original'
else:
    filename = 'Modified'
    datafile = 'Lab1DataNo_3+5.csv'

x, dx, t_na, dt_na, net511, gross511, dnet511, R511, dR511, \
net1270, gross1270, dnet1270, R1270, dR1270, \
x2, dx2, t_ba, dt_ba, net31, gross31, dnet31, R31, dR31,\
net81, gross81, dnet81, R81, dR81, net356, gross356, dnet356, R356, dR356, \
= loadtxt(datafile, unpack=True, skiprows=2, delimiter=',')

Rate_v_X = plt.figure(1)
plt.suptitle("Figures 9-13: Rate of Photon Incidence as a Function of Attenuation Length\n" \
    "for All Measured Energies Without the Data for 2.0 cm Attenuation Lengths")

rate_list = [(31,{'rate':R31,'uncertainty':dR31}),
    (81,{'rate':R81,'uncertainty':dR81}),(356,{'rate':R356,'uncertainty':dR356}),
    (511,{'rate':R511,'uncertainty':dR511}), 
    (1270,{'rate':R1270,'uncertainty':dR1270})]

p0dict = {31:[504,1,0],81:[234,1,0],356:[264,1,0],511:[504,1,0],1270:[80,1,0]}
plotint = 320

residual2cmlist = []

paramlist = []

for item in rate_list:
    if item[0] == item[0]:
        rate = item[1]['rate']
        drate = item[1]['uncertainty']
        p01 = p0dict[item[0]]
        pf1, cov1, info1, mesg1, success1 = optimize.leastsq(residual, p01,
                                    args = (x, rate, drate), full_output=1)
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
        plotint += 1
        if plotint == 225:
            ax = plt.subplot2grid((3,4), (3, 2), colspan=2)
        else:
            ax = Rate_v_X.add_subplot(plotint)
        
        ax.errorbar(x,rate,yerr=drate,xerr=dx,fmt='k.',label='Data')
        if plotfits:
            textfit = '$R(x) = R_0e^{-\\lambda x}$ + B\n' \
              '$R_0 = %.1f \pm %.1f$ Counts/Second \n' \
              '$\\lambda = %.4f \pm %.4f$ cm$^{-1}$\n' \
              '$B = %.2f \pm %.2f$ Counts/Second\n' \
              '$\chi^2= %.2f$ \n' \
              '$N = %i$ (dof) \n' \
              '$\chi^2/N = % .2f$' \
               % (pf1[0], pferr1[0], pf1[1], pferr1[1], pf1[2], pferr1[2], 
                  chisq1, dof1, chisq1/dof1)
            # ax.text(0.4, .85, ("$E_{\\gamma} = $"+str(item[0])+" KeV"), transform=ax.transAxes, fontsize=12,
            #  verticalalignment='top')
            ax.text(0.4, .85, textfit, transform=ax.transAxes, fontsize=12,
             verticalalignment='top')
            
            ax.plot(X,fitfunc(pf1,X),'b-',label='fit')



        plt.savefig(filename+'_Rate_Fits.pdf')
        Rate_v_X.text(0.5, 0.04, 'Attenuation Length (cm)', ha='center')
        Rate_v_X.text(0.04, 0.5, 'Photon Count Rate (counts/second)', 
            va='center', rotation='vertical')
        paramlist.append((item[0],pf1[0], pferr1[0], pf1[1], 
            pferr1[1], pf1[2], pferr1[2], chisq1, dof1, chisq1/dof1))
        if origData:
            residual2cmlist.append(sdresid2cm(pf1,x,rate,drate))

with open(filename+"_Rate_Params.txt",'w+') as csvfile:
    writer = csv.writer(csvfile)
    for tup in paramlist:
        writer.writerow(tup)

if origData:
    print(residual2cmlist)
    totalprob = 1
    for resid in residual2cmlist:
        totalprob *= erfc(resid)
    print(totalprob)
    fig2 = plt.figure(2)
    plt.title("Figure 3: Residuals for 2.0 cm Attenuation Length Data")
    ax1 = fig2.add_subplot(111)
    ax1.bar(range(len(residual2cmlist)),residual2cmlist,color='k')
    ax1.set_xticklabels([0,31, 81, 356,511,1270])
    ax1.set_xlabel("Energy (KeV)")
    ax1.set_ylabel("Residual $\\dfrac{|y-f(x)|}{\\sigma}$")
    plt.savefig("2.0 cm residuals.pdf")


plt.show()



    







