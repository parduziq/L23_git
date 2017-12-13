import os
from params import *
from runSim import runSim
from spiketrain import *
from scipy import io
import matplotlib.pyplot as plt
import time
import numpy as np
from vanRossum import *
import math
import copy



#Import sampled data from 3-way-copula
dat = io.loadmat('/Users/qendresa/Dokumente/sampledata.mat')
sample = np.array(dat['X'])  # 1.EPSP, 2. RC, 3.STP


datestr = time.strftime("%d_%m_%Y")
timestr = time.strftime("%H_%M")

path = "/Users/qendresa/Desktop/L23/Simulation/Syn_dynamics_vRD2_%s"%datestr
if not os.path.exists(path):
    os.makedirs(path)


#Parameter input for the simulations & init Neuron
ninputs = 500
Freq = 0.005
p = params(sample, ninputs, 0.8)

#generate correlation input matrix for spiketrains

c = np.random.choice(p.RC*100, (ninputs, ninputs))

g = np.cov(c)



corr = np.zeros((ninputs, ninputs))
for i in range(ninputs):
    for j in range(i, ninputs):
        corr[i, j] =g[i,j]/ math.sqrt(g[i,i]* g[j,j])
i_lower = np.tril_indices(ninputs, -1)
corr[i_lower] = corr.T[i_lower]

binwidth = 0.1
plt.figure()
plt.subplot(211)
plt.hist(g.flatten(), bins=np.arange(min(g.flatten()), max(g.flatten()) + binwidth, binwidth))
plt.title("Covariance matrix input")

plt.subplot(212)
binwidth = 0.02
plt.hist(corr.flatten(), bins=np.arange(min(corr.flatten()), max(corr.flatten()) + binwidth, binwidth))
plt.title("Correlation matrix input")
plt.tight_layout()
plt.savefig(path+"/CoVCoRmatrix.png")


plt.figure()
plt.subplot(121)
im = plt.imshow(g, cmap='hot')
plt.colorbar()
plt.title("Covariance matrix")

plt.subplot(122)
im = plt.imshow(corr, cmap='hot')
plt.colorbar()
plt.title("Correlation matrix")

plt.savefig(path + '/CovCor.png')


numRuns = 150
globalEPSP = []
globalRC = []
globalPPR = []
final_RC = np.zeros(10000000)
idx = 0

VRD_avgcount = np.zeros((ninputs, ninputs))
Frobeniusnorm = []

for n in range(numRuns):
    data, rank, RC, covSum = gentrain(p, p.nInputs, 1, Freq, g)
    spikes = np.copy(data)
    inputs = [np.where(l == 1)[0] for l in data]
    v,t,s, PPR, EPSP = runSim(p, inputs, rank.astype(int))

    #compute van Rossum distance matrix
    t_VRD_0 = time.time()
    VRD = np.zeros((ninputs, ninputs))
    tau = 0.001
    for k in range(ninputs):
        for l in range(k, ninputs):
            VRD[k, l] = vanRossum(data[k], data[l], tau)
    i_lower = np.tril_indices(ninputs, -1)
    VRD[i_lower] = VRD.T[i_lower]
    t_VRD_1 = time.time()
    print("Computing time: ", t_VRD_1 - t_VRD_0)
    VRD_avgcount = (VRD_avgcount + VRD)

    norm_VRD = VRD_avgcount/np.max(VRD_avgcount)
    norm_VRD = np.ones(norm_VRD.shape)-norm_VRD
    diff = corr - norm_VRD
    Frobeniusnorm.append(np.linalg.norm(diff))

    if n==(numRuns-1):
        VRD_final = VRD_avgcount/np.amax(VRD_avgcount) # normalize it
        VRD_final = np.ones(VRD_final.shape) - VRD_final
        plt.figure()
        im = plt.imshow(VRD_final, cmap='hot')
        plt.colorbar()
        plt.title("Final normalized vanRossum distance %d runs" % (i + 1))
        plt.savefig(path + '/VRD_final_%d_runs.png' % (i+1))

    plt.figure()
    plt.scatter(np.tril(corr).flatten(), np.tril(norm_VRD).flatten())
    plt.axis([-0.2, 0.2, 0.0, 1.0])
    plt.xlabel("Input Correlation")
    plt.ylabel("van Rossum distance")
    plt.savefig(path + "/IC_vRD_%d.png"%(n+1))


    #plt.savefig(path + '/VRD_%dRuns.png' %i)
    activeEPSP=[]
    activeRC=[]
    activePPR =[]
    globactiveIdx = []

    if len(s)>0:
        #for all outputspikes
        outspike = np.zeros((1,1000))
        for i in range(len(s)):
            outspike[0,int(s[i])]=1
            before = int(s[i] - 10) # 10ms before spike
            stim = int(s[i]+1)
            sub_mat = data[:, before:stim] #extract spike mat
            activeIdx = np.where(sub_mat[:,:]==1)[0] #find active inputs
            activeRC.extend(RC[activeIdx])
            activeEPSP.extend((EPSP[activeIdx]*1000))
            activePPR.extend(PPR[activeIdx])
            globactiveIdx.extend(activeIdx.tolist())

        globalEPSP.extend(activeEPSP)
        globalPPR.extend(activePPR)
        globalRC.extend(activeRC)

        #all vRd with respect to output
        for k in range(ninputs):
            d = vanRossum(outspike[0,:], data[k],tau).tolist()
            final_RC[idx] = d
            idx = idx+1


    if ((len(s) >0 and n>1 and n%10 == 0)):
        print("Frobenius difference: ", Frobeniusnorm)
        print("count nonzeros in RC:", np.count_nonzero(final_RC))

        plt.figure()
        binwidth = 0.02
        plt.subplot(411)
        max_edge= max(norm_VRD.flatten())
        min_edge = min(norm_VRD.flatten())
        N = (max_edge - min_edge) / binwidth + 1;
        bin_list = np.linspace(min_edge, max_edge , N)
        plt.hist(norm_VRD.flatten(), bins=bin_list)
        plt.title("All normlized vR-distances")

        plt.subplot(412)
        max_edge= max(norm_VRD[i,:])
        min_edge = min(norm_VRD[i,:])
        N = (max_edge - min_edge) / binwidth + 1;
        bin_list = np.linspace(min_edge, max_edge , N)
        plt.hist(norm_VRD[i,:], bins=bin_list)
        plt.title("Random output and their vR values")

        plt.subplot(413)
        max_edge= max(norm_VRD[n,:])
        min_edge = min(norm_VRD[n,:])
        N = (max_edge - min_edge) / binwidth + 1;
        bin_list = np.linspace(min_edge, max_edge , N)
        plt.hist(norm_VRD[n,:], bins=bin_list)
        plt.title("Random output and their vR values")

        plt.subplot(414)
        pairwise = np.random.choice(norm_VRD.flatten(), 1500)
        max_edge= max(pairwise)
        min_edge = min(pairwise)
        N = (max_edge - min_edge) / binwidth + 1;
        bin_list = np.linspace(min_edge, max_edge , N)
        plt.hist(pairwise, bins=bin_list)
        plt.title("Randomly sampled vR values (pairwise data)")
        plt.tight_layout()
        plt.savefig(path + "/vRvalues_%d.png"%(n+1))

        plt.figure(0)
        plt.plot(np.arange(len(Frobeniusnorm)), Frobeniusnorm)
        plt.title("Forbenius norm of difference matrix (corr - VRD)")
        plt.savefig(path + "/Frob_%d.png"%(n+1))


        plt.figure(1)

        binwidth = 10
        plt.subplot(3,1,1)
        max_edge= max(covSum)
        min_edge = min(covSum)
        N = (max_edge - min_edge) / binwidth + 1;
        bin_list = np.linspace(min_edge, max_edge , N)
        plt.hist(covSum, bins=bin_list)
        plt.title("coVSum values")

        plt.subplot(3,1,2)
        binwidth = 0.1
        max_edge= max(globalEPSP)
        min_edge = min(globalEPSP)
        N = (max_edge - min_edge) / binwidth + 1;
        bin_list = np.linspace(min_edge, max_edge , N)
        plt.hist(globalEPSP, bins=bin_list)
        plt.title("active EPSP-values")

        plt.subplot(3,1,3)
        binwidth = 0.05
        max_edge= max(globalPPR)
        min_edge = min(globalPPR)
        N = (max_edge - min_edge) / binwidth + 1;
        bin_list = np.linspace(min_edge, max_edge , N)
        plt.hist(globalPPR, bins=bin_list)
        plt.title("active PPR")
        plt.tight_layout()
        plt.savefig(path + '/Hist_%dRuns.png' %(n+1))

        # plt.figure()
        # nzRC = final_RC[np.nonzero(final_RC)]
        # norm_RC = np.ones(nzRC.shape) - nzRC/np.amax(nzRC)
        # np.save(path+"/outRC", norm_RC)
        # print("X-y, shape:", norm_RC[n*ninputs:(n*ninputs+ninputs)].shape, covSum.shape)
        # plt.scatter(norm_RC[n*ninputs:(n*ninputs+ninputs)], covSum)
        # plt.xlabel("similarity-measure (vRd)")
        # plt.ylabel("impact in covariance matrix")
        # plt.title("Similarity of in & output spiketrain vs. Correlationinput")
        # plt.savefig(path+ "/Scatter_%d.png"%(n+1))

#plt.show()

#plt.figure(2)
#plt.subplot(2,1,1)
#plt.plot(t,v)
#plt.ylim([-80, 40])

#ax = plt.subplot(2,1,2)
#ax.eventplot(inputs)
#ax.set_xlim([0,1000])
#plt.savefig(path + '/lastRun_%d.png'%numRuns)



