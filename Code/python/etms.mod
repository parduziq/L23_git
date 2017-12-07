NEURON {
	POINT_PROCESS etms
	RANGE e, i
	RANGE R,f
	RANGE tau_1, tau_rec, tau_facil, U, f_in, fac_step, R_0, f_0
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(umho) = (micromho)
}

PARAMETER {
	e = 0 (mV)
	tau_1 = 3 (ms) < 1e-9, 1e9 >
	tau_rec = 100 (ms) < 1e-9, 1e9 >
	tau_facil = 1000 (ms) < 0, 1e9 >
	U = 0.04 (1) < 0, 1 >
	fac_step = 0 (1) < 0, 1 >
	R_0 = 1
	f_0 = 0
}

ASSIGNED {
	v (mV)
	i (nA)
	R
	f
	u
}

STATE {
	g (umho)
}

INITIAL {
	g=0
	R=R_0
	f=f_0
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	i = g*(v - e)
}

DERIVATIVE state {
	g' = -g/tau_1
}

NET_RECEIVE(weight (umho), tsyn (ms)) {
    INITIAL {
        tsyn = t
    }

    :calculate recovery of R, decay of f, calculate u from f
    R = 1 - (1-R) * exp(-(t - tsyn)/tau_rec)
    f = f*exp(-(t - tsyn)/tau_facil)
    u = U + f*(1-U)

    :calculate g
    state_discontinuity(g, g + weight*R*u)

    :update R, f and tsyn
    R = R - u*R
    f = f + fac_step*(1-f)
	tsyn = t
}