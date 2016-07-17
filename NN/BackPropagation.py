
# coding: utf-8

# In[2]:

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# In[17]:

class Neural(object) :
    def __init__(self) :
        self.inputLayer = 3
        self.hiddenLayerOne = 4
        self.hiddenLayerTwo = 4
        self.outputLayer = 1
        self.eta = 0.2    #LEARNING RATE
        self.W1 = np.random.randn(self.hiddenLayerOne,self.inputLayer)
        self.W2 = np.random.randn(self.hiddenLayerTwo,self.hiddenLayerOne)
        self.W3 = np.random.randn(self.outputLayer,self.hiddenLayerTwo)
    def sigmoid(self,z) :
        return 1.0/(1.0+ np.exp(-z))
    def dsigmoid(self,z) :
        return np.dot((self.sigmoid(z)), (1 - self.sigmoid(z)))
    def cost_function(sekf,y,yHat) :
        return np.dot((y - np.transpose(yHat)),np.transpose(y - np.transpose(yHat)))
    def forward_propagation(self,X) :
        self.z2 = np.dot(self.W1,X)
        print "Z2 : ",self.z2
        self.a2 = self.sigmoid(self.z2)
        print "A2 : ",self.a2
        self.z3 = np.dot(self.W2,self.a2)
        print "Z3 : ",self.z3
        self.a3 = self.sigmoid(self.z3)
        print "A3 : ",self.a3
        self.z4 = np.dot(self.W3,self.a3)
        print "Z4 : ",self.z4
        self.a4 = self.sigmoid(self.z4)
        print "A4 : ",self.a4
        self.yHat = np.sum(self.a4)
        print "yHat : ",self.yHat
    def backpropagation(self,Y) :
        self.lasterror = np.dot(np.dot(self.yHat,1.0-self.yHat),self.yHat - Y)
        self.twoerror = np.multiply(np.transpose(self.W3)*self.lasterror,np.multiply(self.a4,1.0-self.a4))
        #print np.shape(x.twoerror)
        self.oneerror = np.multiply(np.transpose(self.W2)*self.lasterror,np.multiply(self.a3,1.0-self.a3))
        self.zeroerror = np.multiply(np.transpose(self.W1)*self.lasterror,np.multiply(self.a2,1.0-self.a2))
        #print self.lasterror,self.twoerror,self.oneerror,self.zeroerror
        return self.gradient_descent()
    def gradient_descent(self) :
        self.W1 = self.W1 + np.transpose((self.zeroerror))
        self.W2 = self.W2 + np.transpose(self.eta*self.oneerror)
        self.W3 = self.W3 + np.transpose(self.eta*self.twoerror)


# In[18]:

n = Neural()
n.forward_propagation(np.array([2.0,3.0,4.0]))


# In[19]:

n.backpropagation(np.array([5]))


# In[ ]:




# In[ ]:



