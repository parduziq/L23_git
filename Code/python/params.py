class params:

    def __init__(self, dat,  nInputs, m):
        self.EPSP = dat[:, 0]/1000
        self.RC = dat[:,1]
        self.PPR = dat[:,2]
        self.m = m  # correlation between RC and EPSP

        ##########
        # inputs #
        ##########

        self.nInputs = nInputs # axons converging onto cell
        self.len = 1000 #time in ms

        ##############
        # morphology #
        ##############

        # size parameters approximated according to wiki pyramidal neuron
        self.soma_nseg = 1
        self.soma_diam = 20
        self.soma_L = 20
        self.soma_Ra = 1
        self.soma_cm = 1.7

        self.dend_nseg = 1
        self.dend_diam = 2
        self.dend_L = 100
        self.dend_Ra = 100
        self.dend_cm = 1.3

        ############
        # synapses #
        ############
        self.nSyn = 1 # synapses per axon
        self.tau_g = 3
        self.e = 0
