'''
created by Gaurav Mitra
'''
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import operator


# In[38]:

class Neural(object) :
    def __init__(self,train) :
        self.inputLayer = 4
        self.hiddenLayerOne = 5
        self.hiddenLayerTwo = 5
        self.outputLayer = 1
        self.eta = 0.15    #LEARNING RATE
        self.W1 = np.random.randn(self.hiddenLayerOne,self.inputLayer)
        self.W2 = np.random.randn(self.hiddenLayerTwo,self.hiddenLayerOne)
        self.W3 = np.random.randn(self.outputLayer,self.hiddenLayerTwo)
        self.training = train
        self.data = Dataset(self.training)
        for i in xrange(self.training) :
            self.Xmatrix = self.data.get_data()[i].get_X()
            self.Ymatrix = self.data.get_data()[i].get_Y()
            for j in xrange(len(self.Ymatrix)):
                self.forward_propagation(np.transpose(self.Xmatrix[[j],:]))
                self.backpropagation(self.Ymatrix[j])
    def sigmoid(self,z) :
        return 1.0/(1.0+ np.exp(-z))
    def dsigmoid(self,z) :
        return np.dot((self.sigmoid(z)), (1 - self.sigmoid(z)))
    def cost_function(sekf,y,yHat) :
        return np.dot((y - np.transpose(yHat)),np.transpose(y - np.transpose(yHat)))
    def forward_propagation(self,X) :
        self.z2 = np.dot(self.W1,X)
        #print "Z2 : ",self.z2
        self.a2 = self.sigmoid(self.z2)
        #print "A2 : ",self.a2
        self.z3 = np.dot(self.W2,self.a2)
        #print "Z3 : ",self.z3
        self.a3 = self.sigmoid(self.z3)
        #print "A3 : ",self.a3
        self.z4 = np.dot(self.W3,self.a3)
        #print "Z4 : ",self.z4
        self.a4 = self.sigmoid(self.z4)
        #print "A4 : ",self.a4
        self.yHat = np.sum(self.a4)
        #print "yHat : ",self.yHat
    def backpropagation(self,Y) :
        self.lasterror = np.dot(np.dot(self.yHat,1.0-self.yHat),self.yHat - Y)
        self.twoerror = np.multiply(np.dot(np.transpose(self.W3),self.lasterror),np.multiply(self.a4,1.0-self.a4))
        #print np.shape(x.twoerror)
        self.oneerror = np.multiply(np.dot(np.transpose(self.W2),self.twoerror),np.multiply(self.a3,1.0-self.a3))
        #self.zeroerror = np.multiply(np.dot(np.transpose(self.W1),self.oneerror),np.multiply(self.a2,1.0-self.a2))
        #print self.lasterror,self.twoerror,self.oneerror,self.zeroerror
        return self.gradient_descent()
    def gradient_descent(self) :
        self.W1 = self.W1 - np.dot(np.transpose((self.oneerror)),self.a2)
        self.W2 = self.W2 - np.dot(np.transpose(self.eta*self.oneerror),self.a3)
        self.W3 = self.W3 - np.transpose(self.eta*self.twoerror)


# In[39]:

class Data(object) :
    def __init__(self) :
        self.g = nx.barabasi_albert_graph(random.randint(100,1000),random.randint(2,7))
        self.degree_centrality = nx.degree_centrality(self.g)
        self.deg = nx.degree_centrality(self.g)
        self.sorted_deg = sorted(self.deg.items(), key=operator.itemgetter(1))
        self.nodes = len(self.g.nodes())
        self.edges = len(self.g.edges())
        self.degree_rank()
        self.degree_dict = self.g.degree()
        self.avg_deg = sum(self.g.degree().values())/float(len(self.g.nodes()))
        #print self.rank
        #print self.degree_dict
        self.form_dataset()
    def degree_rank(self) :
        self.sorted_deg = sorted(self.deg.items(), key=operator.itemgetter(1))
        count = 1
        self.rank = dict()
        self.rank[self.sorted_deg[0][0]] = len(self.sorted_deg)
        for i in xrange(1,len(self.sorted_deg)) :
            if self.sorted_deg[i-1][1] == self.sorted_deg[i][1] :
                self.rank[self.sorted_deg[i][0]] = self.rank[self.sorted_deg[i-1][0]]
                count += 1
            else :
                self.rank[self.sorted_deg[i][0]] = self.rank[self.sorted_deg[i-1][0]] - count
                count = 1
    def form_dataset(self) :
        #print len(self.rank),len(self.degree_dict)
        self.X = np.zeros((self.nodes,4),dtype='float')
        self.Y = np.zeros(self.nodes)
        for i in xrange(self.nodes) :
            self.X[i][0] = self.nodes
            self.X[i][1] = self.edges
            self.X[i][2] = self.avg_deg
            self.X[i][3] = self.degree_dict[i]
            self.Y[i] = self.rank[i]
    def get_X(self) :
        return self.X
    def get_Y(self) :
        return self.Y


# In[40]:

class Dataset(object) :
    def __init__(self,train):
        self.items = train
        self.training_data = list()
        for i in xrange(train):
            data = Data()
            self.training_data.append(data)
    def get_data(self):
        return self.training_data


# In[41]:

n = Neural(10)


# In[45]:

print n.W1


# In[46]:

print n.W2


# In[47]:

print n.W3


# In[ ]:



