import os
from params import *
from runSim import runSim
from spiketrain import *
from scipy import io
import matplotlib.pyplot as plt
import time
import numpy as np


#Import sampled data from 3-way-copula
dat = io.loadmat('/Users/qendresa/Dokumente/sampledata.mat')
sample = np.array(dat['X'])  # 1.EPSP, 2. RC, 3.STP


datestr = time.strftime("%d_%m_%Y")
timestr = time.strftime("%H_%M")

path = "/Users/qendresa/Desktop/L23/Simulation/Syn_dynamics_%s"%datestr
if not os.path.exists(path):
    os.makedirs(path)


#Parameter input for the simulations & init Neuron
ninputs = 500
Freq = 0.005
p = params(sample, ninputs, 0.8)


numRuns = 100
globalEPSP = []
globalRC = []
globalPPR = []

for i in range(numRuns):
    data, rank, RC = gentrain(p, p.nInputs, 1, Freq, 1)
    inputs = [np.where(l == 1)[0] for l in data]
    v,t,s, PPR, EPSP = runSim(p, inputs, rank.astype(int))
    activeEPSP=[]
    activeRC=[]
    activePPR =[]

    for i in range(len(s)):
        before = int(s[i] - 10) # 10ms before spike
        stim = int(s[i]+1)
        sub_mat = data[:, before:stim] #extract spike mat
        activeIdx = np.where(sub_mat[:,:]==1)[0] #find active inputs
        activeRC.append(RC[activeIdx])
        activeEPSP.append((EPSP[activeIdx]*1000))
        activePPR.append(PPR[activeIdx])

    globalEPSP.extend(activeEPSP)
    globalPPR.extend(activePPR)
    globalRC.extend(activeRC)



# Plot Output Trace with spike Matrix and distribution of parameters
plt.figure(1)
plt.subplot(3,1,1)
plt.hist(globalRC)
plt.title("active RC-values")

plt.subplot(3,1,2)
plt.hist(globalEPSP)
plt.title("active EPSP-values")

plt.subplot(3,1,3)
plt.hist(globalPPR)
plt.title("active PPR")
plt.tight_layout()
plt.savefig(path + '/Hist_%dRuns.png' %numRuns)



plt.figure(2)
plt.subplot(2,1,1)
plt.plot(t,v)
plt.ylim([-80, 40])

plt.subplot(2,1,2)
plt.eventplot(inputs)
plt.xlim([0,1000])
#plt.savefig(path + '/lastRun_%d.png'%numRuns)

plt.show()