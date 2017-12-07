from swap import swap

from neuron import h, gui
import numpy as np
import math



def runSim(p, inputs, rank):
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

    # assign EPSP sizes according to ranks of RC and their correlation p.m
    init_EPSP = np.random.choice(p.EPSP, p.nInputs, replace=False) # randomly sampled EPSPs
    sort_EPSP = np.sort(init_EPSP)
    nswaps = int(round((1-p.m)*len(init_EPSP)))
    shuffled_rank = swap(rank, nswaps)
    EPSP = sort_EPSP[shuffled_rank]

    PPR_samp = np.random.choice(p.PPR, p.nInputs)

    for i in xrange(p.nInputs):
        vecStimList.append(h.VecStim())
        PPR = PPR_samp[i]
        tms = h.tms(0.5, sec=dend)
        tms.u_in = 0.5 # set it high enough so below equation does give no error
        tms.tau_facil = -dt/math.log((PPR/(1-tms.u_in)-1)/(1-tms.u_in))

        synList.append(tms)
        netCon = h.NetCon(vecStimList[i], tms)

        netCon.weight[0] = EPSP[i]
        netConList.append(netCon)

        # play input for vecStim
        vecStimList[i].play(h.Vector(inputs[i]))

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

    return np.asarray(v_soma), np.asarray(t), np.asarray(outVec)
