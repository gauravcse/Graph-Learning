'''
    created by Gaurav Mitra

    INSTRUCTIONS :  python Largeba.py 
'''


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import Queue
from math import sqrt
import random

training_data = list()

def Train(graph) :
    training_data.append(graph)

class KNN(object) :
    def __init__(self,n,m,kavg,kmax,deg,K = 10) :
        self.nodes = n
        self.edges = m
        self.average_degree = kavg
        self.neighbours = K
        self.q = Queue.PriorityQueue(self.neighbours)
        self.k_max = kmax
        self.deg = deg
        
    def compute_distance(self,g) :
        return sqrt(((g.average_degree() - self.average_degree)**2) + ((g.get_edges() - self.edges)**2) + ((g.get_nodes() - self.nodes)**2)  + ((g.get_kmax() - self.k_max)**2) )

    def train(self,data) :
        self.training_data = data
        for graph in self.training_data :
            euclidian_distance = self.compute_distance(graph)
            if self.q.full() == False :
                tup = (-1*euclidian_distance,graph)
                self.q.put(tup)
            elif self.q.full() == True :
                tup1 = self.q.get()
                tup2  = (-1*euclidian_distance,graph)
                if (tup1[0] < tup2[0]) :
                    self.q.put(tup2)
                elif (tup1[0] >= tup2[0]) :
                    self.q.put(tup1)

    def test(self) :
        self.prediction = 0.0
        count = 0
        for i in xrange(self.neighbours) :
            add = 0.0
            tup = self.q.get_nowait()
            g = tup[1]
            self.rank = g.get_degreerank()		# DICT : degree : rank
            degree = self.rank.keys()
            difference = list()
            count = 0
            for each in degree :
                difference.append((abs(each - self.deg),count))
                count += 1
            difference.sort()
            local_rank = 0    #FOR CALCULATING RANK (WEIGHTED) WITH RESPECT TO DEGREE
            local_rank = self.rank[degree[difference[0][1]]]
            self.prediction += (local_rank*(i+1))
        return ((float(self.prediction))/(self.neighbours*(self.neighbours+1)/2.0))
		

class BA_Graph(object) :
    def __init__(self,n,m) :
        self.graph = nx.barabasi_albert_graph(n,m)
        self.nodes = n
        self.edges = m
    def average_degree(self) :
        self.deg = sum(self.graph.degree())
        return (self.deg/self.nodes)
    def get_kmin(self) :
        return (min(self.graph.degree()))
    def get_kmax(self) :
        return (max(self.graph.degree()))
    def get_nodes(self) :
        return self.nodes
    def get_edges(self) :
        return len(self.graph.edges())
    def get_degreerank(self) :
        self.degree_count = dict()		# DICTIONARY THAT KEEPS COUNT OF HOW MANY VERTICES ARE THERE OF EACH DEGREE
        for i in xrange(self.nodes) :
            self.degree_count[self.graph.degree(i)] = self.degree_count.get(self.graph.degree(i), 0) + 1
        self.rank = dict()				#STORES THE RANK CORRESPONDING TO EVERY DEGREE
        sum = 1
        for k in sorted(self.degree_count.keys(),reverse = True) :
            self.rank[k] = sum
            sum += self.degree_count[k]
        return self.rank	

if __name__ == "__main__" :
    f = open('datanokmin.txt','a')
    for i in xrange(10000) :
        b = BA_Graph(random.randint(100,1000000), int(np.random.randint(2,6,size = 1)))
        print 'TRAIN : {}'.format(i+1)        
        Train(b)
    for i in xrange(1000) :
        b = BA_Graph(random.randint(100,1000000), int(np.random.randint(2,6,size = 1)))
        while count < 100 :
            node_for_degree = random.randint(0,b.get_nodes())
            t = KNN(b.get_nodes(),b.get_edges(),b.average_degree(),b.get_kmax(),b.graph.degree(node_for_degree),500)
            t.train(training_data)
            rank = t.test()
            print b.graph.degree(node_for_degree)
            s = 'PREDICTED RANK : {0}\nACTUAL RANK :{1}'.format(rank,b.get_degreerank()[b.graph.degree(node_for_degree)])
            print s
            f.write(str(rank)+","+str(b.get_degreerank()[b.graph.degree(node_for_degree)])+"\n")
            count += 1
    f.close()

