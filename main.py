import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy
import EoN as eon
import sklearn
import math


# Read the graph

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

G = nx.read_edgelist("fludata.txt", nodetype=int, data=(("weight", float),))


tmax = 10
beta = 0.01      # transmission rate
mu = 0.5         # recovery rate
initial_infected = 325

tau = beta
gamma = mu

# d = nx.average_degree_connectivity(G)
# print("d", d)


sum = 0
for n in G.nodes():
    sum += G.degree[n]
ave = sum / len(G.nodes())

print("ave", ave)

print(1 /(ave*0.01-0.5))



#



import numpy as np
import matplotlib.pyplot as plt

a = np.random.normal(0,5,10)
# b = np.random.normal(-2,7,100)
data = [a]

plt.boxplot(data) # Or you can use the boxplot from Pandas

y = data

plt.plot( 1, 0, 'r.', alpha=0.2, label ='sdfsdf')

plt.legend()
plt.show()

