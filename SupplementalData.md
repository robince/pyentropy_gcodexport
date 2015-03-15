# Introduction #

Supplementary material for
Ince et. al. (2009) "Python for Information Theoretic Analysis of Neural Data"
Frontiers in Neuroinformatics 3:4, doi: 10.3389/neuro.11.004.2009

Code provided for generation of Figure 1 using pyentropy library. The published figure was produced using version 0.1 of the pyEntropy library.

# Data Generation #

**fp\_trials.py**

```
#
# Supplementary Material for
#
# Ince et. al. (2009) "Python for Information Theoretic Analysis of Neural Data"
# (submitted to Frontiers in Neuroinformatics)
#
# Figure 1 - Data Generation
#

from numpy import array, random, zeros, r_
from pyentropy import DiscreteSystem

# Firing rates for 13 different stimuli, measured in rat somatosensory cortex
# Arabzadeh et. al. (2004) "Whisker vibration information carried by rat 
# barrel cortex neurons" J. Neurosci
rate = array([0.0460, 0.0408, 0.0386, 0.0410, 0.0461, 0.0587, 0.1244, 0.2392, 0.4150, 0.5879, 0.6801, 0.7399, 0.7455])

ns = rate.size
p_spike = rate * 0.4

# Model 8 identical (uncorrelated) neurons
L = 8 

# Run simulutation with 2^'samples' trials 'sims' times. 
# For Figure 1, sims=50, samples=5,6,7,8,9,10,11,12,13
def run_batch(samples, sims):
    Hplugin = zeros((4,sims))
    Hpt = zeros((4,sims))
    Hnsb = zeros((4,sims))
    Hqe = zeros((4,sims))
    Hshrink = zeros((4,sims))
    outs = (Hplugin,Hpt,Hnsb,Hqe,Hshrink)
    
    nt = 2**samples # trials
    print str(sims) + " runs of "+str(nt)+" trials..."
    
    for i in range(sims):
        # independant random spiking
        poisson = random.rand(L,nt,ns)
        data = zeros((L,nt,ns))
        data[ poisson<p_spike ] = 1

        # reshape to input, output vectors
        R = data.T.reshape(nt*ns,L).T.astype(int)
        S = r_[0:ns].repeat(nt).astype(int)

        sys = DiscreteSystem(R,(L,2),S,(1,13))
        sys.calculate_entropies(method='plugin',
                sampling='naive',
                calc=['HX','HXY','HiXY','HshXY'],
                qe_method='plugin',
                methods=['plugin','nsb','pt','qe'])
        sys_shrink = DiscreteSystem(R,(L,2),S,(1,13))
        sys_shrink.calculate_entropies(method='plugin',
                sampling='shrink',
                calc=['HX','HXY','HiXY','HshXY'])

        for v,d in zip(outs,
                        [sys.H_plugin,sys.H_pt,sys.H_nsb,sys.H_qe,sys_shrink.H_plugin]):
            v[0,i] = d['HX']
            v[1,i] = d['HXY']
            v[2,i] = d['HiXY']
            v[3,i] = d['HshXY']

        print ".",

    return outs
```

# Plot #

**fp\_bc\_plot.py**

```
#
# Supplementary Material for
#
# Ince et. al. (2009) "Python for Information Theoretic Analysis of Neural Data"
# (submitted to Frontiers in Neuroinformatics)
#
# Figure 1 - Plot
#

from matplotlib.pylab import figure, subplot
from numpy import load

# Load calculated data - this was generated using the function
# run_batch in fp_trials.py 
data5 = load('data/runbatch5_50.npy')
data6 = load('data/runbatch6_50.npy')
data7 = load('data/runbatch7_50.npy')
data8 = load('data/runbatch8_50.npy')
data9 = load('data/runbatch9_50.npy')
data10 = load('data/runbatch10_50.npy')
data11 = load('data/runbatch11_50.npy')
data12 = load('data/runbatch12_50.npy')
data13 = load('data/runbatch13_50.npy')
data = [data5, data6, data7, data8, data9, data10, data11, data12, data13]
x = [5, 6, 7, 8, 9, 10, 11, 12, 13]

# normal informations
I = []
for d in data:
    I.append(d[:,0,:] - d[:,1,:])
            
# shuffled informations
Ish = []
for d in data:
    Ish.append(d[:,0,:] - d[:,2,:] + d[:,3,:] - d[:,1,:])

f = figure(figsize=[3.54,1.75])

ax1 = subplot(1,2,1)
ax2 = subplot(1,2,2)

# normal informations
#  qe
ax1.errorbar( x, [i[3,:].mean() for i in I], 
            yerr=[i[3,:].std()/2 for i in I], 
            label='QE$$',ls='-',c='k',lw=2.0,elinewidth=3.0,capsize=2.0)
#  plugin
ax1.errorbar( x, [i[0,:].mean() for i in I], 
            yerr=[i[0,:].std()/2 for i in I], 
            label='Plugin$$.',ls=':',c='b',lw=2.0,elinewidth=3.0,capsize=2.0)
#  pt 
ax1.errorbar( x, [i[1,:].mean() for i in I], 
            yerr=[i[1,:].std()/2 for i in I], 
            label='PT$$',ls='-',c='r',lw=2.0,elinewidth=3.0,capsize=2.0)
#  nsb
ax1.errorbar( x, [i[2,:].mean() for i in I], 
            yerr=[i[2,:].std()/2 for i in I], 
            label='NSB$$',ls=':',c='g',lw=2.0,elinewidth=3.0,capsize=2.0)
# shrink
ax1.errorbar( x, [i[4,:].mean() for i in I], 
            yerr=[i[4,:].std()/2 for i in I], 
            label='Shrink$$',ls='-',c='y',lw=2.0,elinewidth=3.0,capsize=2.0)

# shuffled informations
#  qe
ax2.errorbar( x, [i[3,:].mean() for i in Ish], 
            yerr=[i[3,:].std()/2 for i in Ish], 
            label='QE$$',ls='-',c='k',lw=2.0,elinewidth=3.0,capsize=2.0)
#  plugin
ax2.errorbar( x, [i[0,:].mean() for i in Ish], 
            yerr=[i[0,:].std()/2 for i in Ish], 
            label='Plugin$$',ls=':',c='b',lw=2.0,elinewidth=3.0,capsize=2.0)
#  pt 
ax2.errorbar( x, [i[1,:].mean() for i in Ish], 
            yerr=[i[1,:].std()/2 for i in Ish], 
            label='PT$$',ls='-',c='r',lw=2.0,elinewidth=3.0,capsize=2.0)
#  nsb
ax2.errorbar( x, [i[2,:].mean() for i in Ish], 
            yerr=[i[2,:].std()/2 for i in Ish], 
            label='NSB$$',ls=':',c='g',lw=2.0,elinewidth=3.0,capsize=2.0)
# shrink
ax2.errorbar( x, [i[4,:].mean() for i in Ish], 
            yerr=[i[4,:].std()/2 for i in Ish], 
            label='Shrink$$',ls='-',c='y',lw=2.0,elinewidth=3.0,capsize=2.0)

yticks = [0.4,0.6,0.8,1,1.2]
xticks = [6,8,10,12]
xticklabels = ['6$$','8$$','10$$','12$$']
yticklabels = ['0.4$$','0.6$$','0.8$$','1.0$$','1.2$$']
ax1.set_yticklabels(yticklabels)
ax2.set_yticklabels([])

for ax in (ax1,ax2):
    ax.set_ylim([0.38,1.3])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(yticks)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.set_frame_on(False)
    p, = ax.plot([0,0],[0,1], "k-", transform=ax.transAxes)
    p.set_clip_on(False)
    p, = ax.plot([0,1],[0,0], "k-", transform=ax.transAxes)
    p.set_clip_on(False)

l = ax2.legend(handlelen=0.2)

ax1.set_ylabel(r'Information (bits)$$')
#ax1.set_xlabel('Log$_2$(trials/stim)')
f.text(0.53,0.01,r'Log$_2$(trials/stim)',ha='center',family='sans-serif')

ax1.set_position([0.125,0.18,0.38,0.7])
ax2.set_position([0.55,0.18,0.38,0.7])

f.text(0.09,0.92,'A',weight='bold',ha='center')
f.text(0.5125,0.92, 'B',weight='bold',ha='center')

f.text(0.37,0.55,'$\mathbf{I(S;R)}$',weight='bold',ha='center')
f.text(0.63,0.55, '$\mathbf{I_{sh}(S;R)}$',weight='bold',ha='center')

f.show()
```