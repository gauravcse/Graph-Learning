
# coding: utf-8

# In[2]:

#VARIATION OF NODE NUMBER WITH AVERAGE DEGREE


# In[53]:

import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
get_ipython().magic(u'matplotlib inline')


# In[54]:

x = np.zeros(1000)
y = np.zeros(1000)
plt.title("Nodes vs Avg Degree")
plt.xlabel("Nodes")
plt.ylabel("Degree Average")
for i in range(1000) :
    n = random.randint(20,10000)
    G = nx.barabasi_albert_graph(n,random.randint(2,7))
    avg_deg = float(sum(G.degree()))/float(n)
    x[i] = n
    y[i] = avg_deg
plt.scatter(x,y)


# In[55]:

points = np.column_stack((x,y))
np.savetxt("results.csv",points,delimiter=",")


# In[56]:

regr = linear_model.LinearRegression()


# In[57]:

x = x.reshape(-1,1)
regr.fit(x,y)


# In[58]:

print regr.coef_


# In[59]:

print "Mean ERROR : ",(np.mean((regr.predict(x) - y)**2))**0.5


# In[63]:

plt.scatter(x,y,color="blue")
plt.plot(x, regr.predict(x), color='red',
         linewidth=0.5)
plt.show()


# In[ ]:



