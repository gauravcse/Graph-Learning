
# coding: utf-8

# In[1]:

import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
get_ipython().magic(u'matplotlib inline')


# In[5]:

def generate(p) : 
    x = np.zeros(500)
    y = np.zeros(500)
    for i in range(500) :
        n = random.randint(50,1000)
        G = nx.barabasi_albert_graph(n,p)
        avg_deg = float(sum(G.degree()))/float(n)
        x[i] = len(G.edges())
        y[i] = avg_deg
    return x,y


# In[6]:

points = np.zeros((500,2))
for i in xrange(500) :
    p = random.randint(3,40)
    print "done"
    x,y = generate(p)
    slope = Regression(x,y)
    points[i][0],points[i][1] = p,slope
    print "Done"
np.savetxt("pref_slope_1.csv",points,delimiter=",")


# In[3]:

def Regression(x,y) :
    regr = linear_model.LinearRegression()
    x = x.reshape(-1,1)
    regr.fit(x,y)
    print regr.coef_
    return regr.coef_


# In[ ]:



