import os
from params import *
from runSim import runSim
from spiketrain import *
from scipy import io
import matplotlib.pyplot as plt
import time
import numpy as np
from vanRossum import *


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

#generate correlation input matrix for spiketrains
d = np.random.choice(p.RC, (ninputs, ninputs), replace=False)
c = np.dot(d, d.T)

#turn covariance matrix c into correlation matrix
std_ = np.sqrt(np.diag(c))
corr = c / np.outer(std_, std_)


numRuns = 1
globalEPSP = []
globalRC = []
globalPPR = []


VRD_avgcount = np.zeros((ninputs, ninputs))

for i in range(numRuns):
    data, rank, RC = gentrain(p, p.nInputs, 1, Freq, c)
    inputs = [np.where(l == 1)[0] for l in data]
    v,t,s, PPR, EPSP = runSim(p, inputs, rank.astype(int))

    #compute van Rossum distance matrix
    t_VRD_0 = time.time()
    VRD = np.zeros((ninputs, ninputs))
    tau = 0.001
    for i in range(ninputs):
        for j in range(i, ninputs):
            VRD[i, j] = vanRossum(data[i], data[j], tau)
    i_lower = np.tril_indices(ninputs, -1)
    VRD[i_lower] = VRD.T[i_lower]
    t_VRD_1 = time.time()
    print("Computing time: ", t_VRD_1 - t_VRD_0)
    VRD_avgcount = (VRD_avgcount + VRD)/float(i)

    #plt.savefig(path + '/VRD_%dRuns.png' %i)

    activeEPSP=[]
    activeRC=[]
    activePPR =[]

    for i in range(len(s)):
        before = int(s[i] - 10) # 10ms before spike
        stim = int(s[i]+1)
        sub_mat = data[:, before:stim] #extract spike mat
        activeIdx = np.where(sub_mat[:,:]==1)[0] #find active inputs
        activeRC.extend(RC[activeIdx])
        activeEPSP.extend((EPSP[activeIdx]*1000))
        activePPR.extend(PPR[activeIdx])

    globalEPSP.extend(activeEPSP)
    globalPPR.extend(activePPR)
    globalRC.extend(activeRC)



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

#plt.show()

plt.figure(2)
plt.subplot(2,1,1)
plt.plot(t,v)
plt.ylim([-80, 40])

ax = plt.subplot(2,1,2)
ax.eventplot(inputs)
ax.set_xlim([0,1000])
#plt.savefig(path + '/lastRun_%d.png'%numRuns)

# Plot Output Trace with spike Matrix and distribution of parameters


#d, fig = vanRossum(data[1], data[2], tau = 0.008)


#plt.show()