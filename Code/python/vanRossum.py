import numpy as np
import matplotlib.pyplot as plt

def vanRossum(x,y,tau):

    t = np.asarray(range(len(x)))/1000.0
    eps = np.exp(-t/tau)
    x_conv = np.convolve(x,eps)
    x_conv = x_conv[0:len(x)-1]
    y_conv = np.convolve(y,eps)
    y_conv = y_conv[0:len(y)-1]

    d = 1/(tau*1000) * np.sum(np.square(x_conv-y_conv))
    #fig = plt.figure(1)
    #plt.subplot(3,1,1)
    #plt.plot(x_conv)
    #plt.title("convoluted spiketrain 1")
    #plt.subplot(3,1,2)
    #plt.plot(x)
    #plt.title("spiketrain 1")
    #plt.subplot(3,1,3)
    #plt.plot(y)
    #plt.title("spiketrain 2")
    #plt.tight_layout()

    return d