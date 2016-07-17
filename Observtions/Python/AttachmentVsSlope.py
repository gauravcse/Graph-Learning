
# coding: utf-8

# In[2]:

import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import pandas as pd
get_ipython().magic(u'matplotlib inline')


# In[3]:

data = pd.read_csv("pref_slope.csv")
data = data.as_matrix()


# In[5]:

x,y = data[:,[0]],data[:,[1]]


# In[10]:

plt.xlabel("Prefferential Attachment")
plt.ylabel("Slope")
plt.scatter(x,y)
plt.show()


# In[28]:




# In[ ]:




# In[ ]:



