from scipy.stats import rankdata
import numpy as np

def gentrain(p, ninputs, avg_display, H, c):

    #dat = io.loadmat('/Users/qendresa/Desktop/L23/Code/sampledata.mat')
    #sample = np.array(dat['X'])
    #rc = sample[:, 1] # second column
    #ninputs = 50
    #avg_display =1
    #div =1
    #H=0.005


    #sample randomly whole matrix  ## one diagonal
    #for i in xrange(ninputs):
        #d[(i+1):, i] = rc[idx: idx+ninputs-(i+1)] #lower triangular
     #   d[(i+1):, i] = np.random.choice(rc, ninputs - (i + 1))  # upper triangular
     #   d[i, (i+1):] = np.random.choice(rc, ninputs-(i+1)) #upper triangular
     #   idx = idx+1
        #idx = idx+ abs(ninputs - (i+1))





    #uniform_c = np.ones((ninputs, ninputs))/ 1000000000000
    corrSum = np.sum(c, axis = 0)
    RC = np.mean(c, axis = 0)
    rank = rankdata(corrSum) - 1


    #generate spike trains
    T = 1000
    rateMat = np.random.multivariate_normal(np.ones(ninputs)*H, c/100000000, T)
    spikeMat = np.random.poisson(rateMat.T)
    s = (spikeMat != 0).astype(int)


    # check average firing rates
    if avg_display == 0:
     avg = np.sum(s, axis=1)
     print( 'Sum of spikes:', avg, 'Mean firing rate:', np.mean(avg))

    return s, rank, RC, corrSum


"""

d2 = np.diag(v)
for i in range(ninputs):
    d2[(i+1):, i] = np.random.choice(rc, ninputs-(i+1))
c2 = abs(d2 + np.tril(d2, k= -1).T)
corrSum2 = np.sum(c2, axis = 0)
#c2 = c2**2

fig2 = plt.figure(figsize=(4,6))
plt.subplot(2,1,1)
plt.imshow(c2, cmap = "afmhot")
plt.title('Columnwise correlation of inputs')
plt.colorbar()

plt.subplot(2,1,2)
plt.hist(corrSum2, bins=ninputs/10)
plt.title('Distribution of correlation-sum')
plt.xlabel('Sum of correlation')
fig2.tight_layout()

plt.show()
"""
