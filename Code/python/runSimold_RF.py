#import sys
#sys.path.append("NEURON-7.4/nrn/lib/python")

from neuron import h, gui
import numpy as np
import random as rand
import math
from scipy import io

def runSim(p, inputs):
    ####################
    #    MORPHOLOGY    #
    ####################

    # create soma with dendrite
    soma = h.Section(name='soma')
    soma.insert('hh')
    soma.nseg = p.soma_nseg
    soma.diam = p.soma_diam
    soma.L = p.soma_L
    soma.Ra = p.soma_Ra
    soma.cm = p.soma_cm

    dend = h.Section()
    dend.nseg = p.dend_nseg
    dend.diam = p.dend_diam
    dend.L = p.dend_L
    dend.Ra = p.dend_Ra
    dend.cm = p.dend_cm
    dend.insert('pas')
    dend.connect(soma, 0, 0)

    ########################
    #    RANDOM CURRENT    #
    ########################

    # insert current to drive cell close to threshold
    # Gfluct mechanism taken from Rudolph M. and Destexhe A. 2006

    #noise = h.Gfluct(0.5, sec=soma)
    #noise.g_e0 = 0.01
    #noise.g_i0 = 0.01
    #noise.std_e = 0.0005
    #noise.std_i = 0.0005
    #noise.new_seed(np.random.random_integers(1, 10000))

    ################################
    #   L23 INPUTS AND SYNAPSES    #
    ################################

    # set up synapses for all inputs and connect to different NetCons
    nSyn = p.nSyn
    vecStimList = []
    # append objects of all inputs to the same lists
    synList = []
    netConList = []
    dt = 20 #ms
    randList = [] # list of random inputs sampled

    for i in range(p.nInputs):
        vecStimList.append(h.VecStim())
        for j in range(nSyn): # for each synapse sample dynamics
            #k = rand.randint(0, 9999)
            #randList.append(k)
            randList = np.arange(0,2000, 1)
            PPR = p.PPR[j]
            tms = h.tms(0.5, sec=dend)
            tms.u_in = 0.2 # set it high enough so below equation does give no error
            tms.tau_facil = -dt/math.log((PPR/(1-tms.u_in)-1)/(1-tms.u_in))


            synList.append(tms)
            netCon = h.NetCon(vecStimList[i], tms)

            netCon.weight[0] = p.EPSP[j]
            netConList.append(netCon)

        # play input for vecStim
        vecStimList[i].play(h.Vector(inputs[i]))

    # generate RF based on sampled RC
    nSamp = len(randList)
    x = np.random.random(nSamp)*10  # set x values randomly between 0 and 0.1
    y = np.ones(nSamp)

    # let RC = x^2 + y^2
    sampRC = p.RC[randList]*100 #corresponding RC values, scale it up to get meaningful x-y values
    for k in range(nSamp):
        if sampRC[k] <= 0.0:  # if correlations are negative
            y[k] = -math.sqrt(abs(-sampRC[k] - x[k] ** 2))
            x[k] = -x[k]
        else:
            y[k] = math.sqrt(abs(sampRC[k] - x[k] ** 2))

    corV = np.corrcoef(np.vstack((x, y)).T)
    io.savemat('/Users/qendresa/Desktop/L23/matlab/RFC_correlations.mat', mdict = {'corV':corV})
    #print(corV)
    coV2 = np.cov(np.vstack((x, y)).T)
    io.savemat('/Users/qendresa/Desktop/L23/matlab/RFC_covariances.mat', mdict={'coV': coV2})

    #print('coV2 = ', corV)

    ################
    #      RUN     #
    ################

    # set up recording variables
    v_soma = h.Vector()
    v_soma.record(soma(0.5)._ref_v)
    t = h.Vector()
    t.record(h._ref_t)

    # record spikes
    h('objref nil, outNetCon')
    h('outNetCon = new NetCon(&v(.5),nil)')
    outNetCon = h.outNetCon
    outVec = h.Vector()
    outNetCon.threshold = -20
    outNetCon.record(outVec)

    # run simulation
    h.tstop = p.len
    h.run()

    return np.asarray(v_soma), np.asarray(t), np.asarray(outVec), PPR
