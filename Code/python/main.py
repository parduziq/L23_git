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


#Parameter input for the simulations
corr = [0.1, 0.3, 0.5, 0.7, 0.9]

n_inputs = np.linspace(50,500, 6).astype(int)
Freq = np.linspace(0.001, 0.015, 8)
avgspike = np.zeros ((len(n_inputs), len(Freq)))
numRuns = 100

datestr = time.strftime("%d_%m_%Y-%H%_M")
timestr = time.strftime("%H_%M")

path = "/Users/qendresa/Desktop/L23/Simulation/Freq_ninputs_%s"%datestr
if not os.path.exists(path):
    os.makedirs(path)

for j in xrange(numRuns):
    spikeCount = []
    for k in np.nditer(n_inputs):
        for i in np.nditer(Freq):
            p = params(sample, k, 0.8)

            #t_spike_0 = time.time()
            data, rank = gentrain(p, p.nInputs, 1, i, 1)
            #t_spike_1 = time.time()
            #print("Generate spike_trains:", t_spike_1 - t_spike_0)

            inputs = [np.where(l == 1)[0] for l in data]

            #fig = plt.figure()
            #plt.subplot(121)
            #plt.eventplot(inputs1)
            #plt.xlim(0, 1000)
            #plt.ylim(1, p.nInputs * p.nSyn)
            #plt.subplot(122)
            #plt.eventplot(inputs)
            #plt.xlim(0, 1000)

            #plt.show()

            t_runSim_0 = time.time()
            v,t,s = runSim(p, inputs, rank.astype(int))
            t_runSim_1 = time.time()
            print("Simulation time:", t_runSim_1-t_runSim_0)

            spikeCount.append(len(s))


    t_rest_0 = time.time()
    avgspike = avgspike + np.asarray(spikeCount, dtype=float).reshape((len(n_inputs),len(Freq)))
      # print("updated count:", avgspike)
    print("avgd count:", (avgspike / (j + 1)))
    plt.figure()
    im = plt.imshow(avgspike / (j + 1), cmap='hot')
    ax = plt.gca()
    ax.set_xticks(np.arange(len(Freq)))
    ax.set_yticks((np.arange(len(n_inputs))))
    ax.set_xticklabels(np.array(Freq) * 1000)
    ax.set_yticklabels(np.array(n_inputs))
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    plt.colorbar()
    plt.title("Avg spike count after %d runs" % (j + 1))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Number of Inputs")
    plt.savefig(path + '/avgCount_%d-%s.png' %(j,timestr))
    plt.close()
    t_rest_1 = time.time()
    print("Generate/Save images:", t_rest_1 - t_rest_0)