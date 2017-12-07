NEURON {
	POINT_PROCESS tms
	RANGE e,i,tau_1
	RANGE R,u
	RANGE tau_rec, tau_facil, u_in
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
	u_in = 0.04 (1) < 0, 1 >
}

ASSIGNED {
	v (mV)
	i (nA)
	R
	u
}

STATE {
	g (umho)
}

INITIAL {
	g=0
	R=1
	u=0
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

    :calculate recovery of R
    R = 1 - (1-R) * exp(-(t - tsyn)/tau_rec)

	: calculate decay of u and increase before event (since it drops to 0)
	if (tau_facil > 0) {
		u = u*exp(-(t - tsyn)/tau_facil)
		u = u + u_in*(1-u)
	} else {
		u = u_in
	}

    :calculate g from current R and u
    state_discontinuity(g, g + weight*R*u)

    :calculate new R and update tsyn
    R = R - u*R
	tsyn = t
}