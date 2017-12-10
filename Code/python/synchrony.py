import numpy as np
from matplotlib.pyplot import *
from spikes import *
from params import *
import os
import pickle


def calcDistance(inputs,p,method):
    tau = 0.01

    # transform spiketimes to spike trains
    trains = np.zeros((p.nInputs,p.len))
    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            if inputs[i][j]-1 < p.len:  # might be bigger bc of bursting
                trains[i,inputs[i][j]-1] = 1

    distances = []
    for i in range(len(inputs)):
        for j in range(i+1,len(inputs)):
            # restrict measure to stim part
            if method is 'vr':
                #distances.append(vanRossum(trains[i],trains[j],tau))
                distances.append(vanRossum(trains[i][p.spontLen:p.spontLen+100],trains[j][p.spontLen:p.spontLen+100],tau))
            if method is 'cc':
                #distances.append(normCC(trains[i],trains[j],10))
                distances.append(normCC(trains[i][p.spontLen:p.spontLen+100],trains[j][p.spontLen:p.spontLen+100],10))
                # change tau...?

    return np.mean(distances)

def vanRossum(x,y,tau):

    t = np.asarray(range(len(x)))/1000.0
    eps = np.exp(-t/tau)
    x_conv = np.convolve(x,eps)
    x_conv = x_conv[0:len(x)-1]
    y_conv = np.convolve(y,eps)
    y_conv = y_conv[0:len(y)-1]

    d = 1/(tau*1000) * np.sum(np.square(x_conv-y_conv))
    # subplot(3,1,1)
    # plot(np.square(x_conv-y_conv))
    # subplot(3,1,2)
    # plot(x)
    # subplot(3,1,3)
    # plot(y)
    # show()
    return d

def normCC(x,y,tau):

    spikeIdx = np.where(x == 1)[0]
    count = 0
    d = 0
    if len(spikeIdx) > 0:
        for i in np.nditer(spikeIdx):
            start = int(i - np.floor(tau/2))
            start = max(start,0)
            stop = int(i + np.ceil(tau/2))+1
            stop = min(stop,len(y)-1)
            count += np.sum(y[start:stop])
        # d = count/np.sqrt((np.sum(x)**2+np.sum(y)**2)/2)
        d = count/((np.sum(x)**2+np.sum(y)**2)/2) # did not divide by the sqrt to avoid observed correlation

    return d



def buildSynBib(p):

    ibib =[]
    ds = []
    for i in range(200):
        inputs = createInputs(p)
        ibib.append(inputs)
        ds.append(calcDistance(inputs,p,'cc'))

    return ds,ibib



def chooseInputs(p,run):

    if p.newInputs:
        ds,ibib = buildSynBib(p)

        min = np.min(ds)
        max = np.max(ds)
        #cut off min and max to avoid choosing outliers
        dVals = np.arange(min,max,(max-min)/(p.synRange+1))
        dVals = dVals[1:]
        out = []
        dVals_chosen = []

        for i in range(len(dVals)):
            idx = np.argmin(abs(ds-dVals[i]))
            out.append(ibib[idx])
            dVals_chosen.append(ds[idx])

        # save output
        pstr = str(int(p.pBurst*100)) + '_' + str(run)
        with open(p.inPath + '/' + pstr, 'wb') as f:
            pickle.dump(out, f)


    else:
        # load output from saved file
        pstr = str(int(p.pBurst*100)) + '_' + str(run)
        print(p.inPath + '/' + pstr)
        with open(p.inPath + '/' + pstr, 'rb') as f:
            out = pickle.load(f)


    return out


def checkRateSynCorr():

    p = params(1)
    ds,ibib = buildSynBib(p)
    ns = [sum([len(i) for i in j]) for j in ibib]
    plot(ds,ns,'+')
    show()


def buildInputBib(run):
    bib = []
    brs = []
    synVals = []
    p = params()
    p.len = 500
    for i in range(10):
        for j in range(10):
            rateVec = np.zeros((p.len,1))
            rateVec[:] = (i+1)*2
            rateVec[200:300] = 50
            for k in range(10):
                pBurst = (j+1.0)/10.0
                inputs = createInputs(rateVec,pBurst,p.nInputs)
                bib.append(inputs)
                brs.append(calcBR(inputs))
                synVals.append(calcSyn(inputs,p.len))

    with open('/home/benjamin/Code/PycharmProjects/TCS/data/sync/newInputs/' + str(run), 'wb') as f:
        pickle.dump([bib,brs,synVals], f)

    return bib


def calcBR(inputs):

    nBursts = 0

    for i in range(len(inputs)):
        train = np.atleast_1d(inputs[i])
        j = 0
        while j < len(train)-2:
            count = 0
            if j is 0:
                isi = train[0]
            else:
                isi = train[j] - train[j-1]
            if isi > 100:
                window = train[j:j+10]
                postIsis = np.diff(window)
                k = 0
                flag = 1
                while flag and postIsis[k] < 5 :
                    count += 1
                    k += 1
                    if k > len(postIsis)-1:
                        flag = 0
                if count > 1:
                    nBursts += 1
            j += count + 1

        br = float(nBursts) / np.sum([np.size(i) for i in inputs])

    return br



def calcSyn(inputs,stimLen):
    # change tau...?
    tau = 10

    # transform spiketimes to spike trains
    trains = np.zeros((len(inputs),stimLen))
    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            if inputs[i][j]-1 < stimLen:  # might be bigger bc of bursting
                trains[i,inputs[i][j]-1] = 1

    distances = []
    for i in range(len(inputs)):
        for j in range(i+1,len(inputs)):
            # restrict measure to stim part
            distances.append(normCC(trains[i][200:300],trains[j][200:300],tau))

    return np.mean(distances)




def chooseIns(run):


    with open('/home/benjamin/Code/PycharmProjects/TCS/data/sync/newInputs/' + str(run), 'rb') as f:
        data = pickle.load(f)
    bib,brs,synVals = data


    # minB = np.min(brs)
    # maxB = np.max(brs)
    # minS = np.min(synVals)
    # maxS = np.max(synVals)

    # use percentiles to get rid of outliers
    minB = np.percentile(brs,10)
    maxB = np.percentile(brs,90)
    minS = np.percentile(synVals,10)
    maxS = np.percentile(synVals,90)
    #cut off min and max to avoid choosing outliers
    bVals = np.arange(minB,maxB,(maxB-minB)/10)
    sVals = np.arange(minS,maxS,(maxS-minS)/10)



    # take input wth minimum euclidean distance
    out = []
    for i,s in enumerate(sVals):
        out.append([])
        for j,b in enumerate(bVals):
            ds = []
            for k in range(len(brs)):
                #dist = np.linalg.norm(np.array([brs[k],synVals[k]])- np.array([b,s]))
                dist = np.sqrt( np.square( (brs[k] - b) / (maxB-minB) ) +  np.square( (synVals[k] - s) / (maxS -minS) ) )
                ds.append(dist)
            ind = np.argmin(ds)
            # figure()
            # plot(synVals,brs,'+')
            # plot(s,b,'o')
            # plot(synVals[ind],brs[ind],'x')
            # show()
            out[i].append(bib[ind])


    return out
