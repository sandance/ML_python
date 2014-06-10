import matplotlib.pyplot as plt
import numpy as np
x = np.arange(1,5)

plt.plot(x,x*1.5,label='Normal')
plt.plot(x,x*3.0,label='Fast')
plt.plot(x,x/3.0,label='Slow')

plt.grid(True)

plt.title('Sample growth of measure')
plt.xlabel('Samples')
plt.ylabel('Values measured')


plt.legend(loc='upper left')
plt.show()


