import numpy as np
import matplotlib.pyplot as plt

plt.figure(1) # The first figure
plt.subplot(211) # the first subplot in the first figure
plt.plot([1,2,3])

plt.subplot(212) # Second subplot in the first figure
plt.plot([4,5,6])


plt.figure(2) # A second figure
plt.plot([4,5,6])

plt.figure(1) # figure 1 current, subplot(212) still current
plt.subplot(211)
plt.title('Easy as 1 ,2,3')
