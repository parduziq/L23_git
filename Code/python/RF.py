import numpy as np
import matplotlib.pyplot as plt
import math

d = np.random.gamma(0.1, 0.5, 1000)
x = np.random.random(1000)/100
y = np.ones(1000)

## set x negative
for k in range(len(d)):
    if d[k] >=0.8:
        #print(d[k])
        d[k] = np.random.uniform(-0.1, 0)
        #print('new d[%i]'%k,'= %f' %d[k])
        y[k] = -math.sqrt(abs(-d[k] - x[k]**2))
        x[k] = -x[k]
    else:
        y[k] = math.sqrt(abs(d[k] - x[k]**2))

#print(np.var(x), np.var(y))
dat = np.vstack((x, y)).T
corV = np.corrcoef(np.vstack((x, y)))
print(corV)





plt.figure(1)
plt.subplot(121)
plt.scatter(x,y)
plt.title('x-y Receptive Field')
plt.subplot(122)
plt.hist(d, bins = 30)
plt.title('d - distance')
plt.show()



#y = np.sqrt(d - x)

#plt.scatter(x,y)
#plt.show()
#plt.hist(d, bins=30)
#plt.show()