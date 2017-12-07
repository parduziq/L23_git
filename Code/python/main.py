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


datestr = time.strftime("%d_%m_%Y-%H%_M")
timestr = time.strftime("%H_%M")

path = "/Users/qendresa/Desktop/L23/Simulation/Syn_dynamics_%s"%datestr
if not os.path.exists(path):
    os.makedirs(path)


#Parameter input for the simulations
n_inputs = 400
Freq = 0.005
numRuns = 100

p = params(sample, n_inputs, 0.8)
data, rank, RC = gentrain(p, p.nInputs, 1, Freq, 1)
inputs = [np.where(l == 1)[0] for l in data]
v,t,s, PPR, EPSP = runSim(p, inputs, rank.astype(int))

act_EPSP=[]
act_RC=[]
act_PPR =[]

for i in range(len(s)):
    before = int(s[i] - 10) # 10ms before spike
    stim = int(s[i]+1)
    sub_mat = data[:, before:stim] #extract spike mat
    act_idx = np.where(sub_mat[:,:]==1)[0] #find active inputs
    act_RC.append(RC[act_idx])
    act_EPSP.append((EPSP[act_idx]))
    act_PPR.append(PPR[act_idx])



# Plot Output Trace with spike Matrix and distribution of parameters
plt.figure(1)
plt.subplot(3,1,1)
plt.hist(act_RC)
plt.title("active RC-values")

plt.subplot(3,1,2)
plt.hist(act_EPSP)
plt.title("active EPSP-values")

plt.subplot(3,1,3)
plt.hist(act_PPR)
plt.title("active PPR")
plt.tight_layout()



plt.figure(2)
plt.subplot(2,1,1)
plt.plot(t,v)
plt.ylim([-80, 40])

plt.subplot(2,1,2)
plt.eventplot(inputs)
plt.xlim([0,1000])
plt.show()
