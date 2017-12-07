from params import *
import numpy as np
from scipy import io

dat = io.loadmat('/Users/qendresa/Desktop/L23/matlab/sampledata.mat')
sample = np.array(dat['X'])

for i in range(10):
    p = params(sample, 1500, 0.8)

ninputs = 100

rc = p.RC
v = np.ones(ninputs)
d = np.diag(v)
idx = 0

# sample randomly whole matrix
for i in xrange(ninputs):
  d[(i+1):, i] = rc[idx: idx+ninputs-(i+1)] #lower triangular
  d[i, (i+1):] = np.random.choice(rc, ninputs-(i+1)) #upper triangular
  idx = idx+ abs(ninputs - (i+1))

c = np.dot(d, d.T)

H = 0.005
T = 1000
div = 1

rateMat = np.random.multivariate_normal(np.ones(ninputs)*H, c/1000000000000, T)
rateMat2 = np.repeat(rateMat, div, axis = 1)
spikeMat = np.random.poisson(rateMat2.T)
s = (spikeMat != 0).astype(int)


np.save("/Users/qendresa/Desktop/L23/F_%d"%H, s)


