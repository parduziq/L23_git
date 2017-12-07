from scipy.stats import rankdata
from CV_matrices import *
import time
from scipy import io

def gentrain(p, ninputs, avg_display, H, div=1):

    #dat = io.loadmat('/Users/qendresa/Desktop/L23/Code/sampledata.mat')
    #sample = np.array(dat['X'])
    #rc = sample[:, 1] # second column
    #ninputs = 50
    #avg_display =1
    #div =1
    #H=0.005

    rc = p.RC
    v = np.ones(ninputs)
    d = np.diag(v)

    idx = 0

    #sample randomly whole matrix
    #for i in xrange(ninputs):
        #d[(i+1):, i] = rc[idx: idx+ninputs-(i+1)] #lower triangular
     #   d[(i+1):, i] = np.random.choice(rc, ninputs - (i + 1))  # upper triangular
     #   d[i, (i+1):] = np.random.choice(rc, ninputs-(i+1)) #upper triangular
     #   idx = idx+1
        #idx = idx+ abs(ninputs - (i+1))
    d = np.random.choice(rc, (ninputs, ninputs), replace=False)
    c = np.dot(d, d.T)



    #uniform_c = np.ones((ninputs, ninputs))/ 1000000000000
    corrSum = np.sum(c, axis = 0)
    rank = rankdata(corrSum) - 1


    #generate spike trains
    T = 1000/div
    #t_sample_0 = time.time()
    rateMat = np.random.multivariate_normal(np.ones(ninputs)*H, c/100000000000, T)

    #t_sample_1 = time.time()
    #rateMat2 = np.repeat(rateMat, div, axis = 0)

    #print("Generate Ratematrix", t_sample_1-t_sample_0)
    #t_spike_0 = time.time()
    spikeMat = np.random.poisson(rateMat.T)
    s = (spikeMat != 0).astype(int)

    #t_spike_1 = time.time()
    #print("Generate spike matrix", t_spike_1-t_spike_0)

    # check average firing rates
    if avg_display == 0:
     avg = np.sum(s, axis=1)
     print( 'Sum of spikes:', avg, 'Mean firing rate:', np.mean(avg))

    return s, rank



