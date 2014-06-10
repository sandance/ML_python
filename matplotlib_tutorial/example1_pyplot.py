import numpy as np
from numpy import *
import matplotlib.pyplot as plt

#plt.plot([1,2,3,4],[1,4,9,16])
# Adding optional third agument which indicares color and line type of the plot
plt.plot([1,2,3,4],[1,4,9,16], 'ro')
plt.axis([0,6,0,20])
plt.ylabel('some numbers')
plt.show()

