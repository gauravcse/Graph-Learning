import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import Queue
from math import sqrt

training_data = list()

def Train(graph) :
	training_data.append(graph)

class KNN(object) :
	def __init__(self,n,m,kavg,kmin,kmax,deg,K = 10) :
		self.nodes = n
		self.edges = m
		self.average_degree = kavg
		self.neighbours = K
		self.q = Queue.PriorityQueue(self.neighbours)
		self.k_min = kmin
		self.k_max = kmax
		self.deg = deg
		
	def compute_distance(self,g) :
		return sqrt(((g.average_degree() - self.average_degree)**2) + ((g.get_edges() - self.edges)**2) + ((g.get_nodes() - self.nodes)**2) + ((g.get_kmin() - self.k_min)**2) + ((g.get_kmax() - self.k_max)**2) )

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
		print 'SELF DEGREE : {}\n'.format(self.deg)
		for i in xrange(self.neighbours) :
			add = 0.0
			tup = self.q.get_nowait()
			g = tup[1]
			print g.get_degreerank()
			self.rank = g.get_degreerank()		# DICT : degree : rank
			'''for key in sorted(self.rank.keys(),reverse = True) :
				self.rank[key] = add
				add += g.degree_count[key]'''
			degree = self.rank.keys()
			'''for key in self.rank.keys() :
				if abs(key - self.deg) < minimum :
					minimum = key'''
			for each in degree :
				if (each - self.deg) == 0 :
					add = self.deg
					count += 1
			print 'MIN : {}\n'.format(add)
			self.prediction += self.rank[add]
		return ((float(self.prediction))/(count))
		

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
		return self.edges
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
	for i in xrange(500) :
		b = BA_Graph(i+50, int(np.random.randint(2,5,size = 1)))
		Train(b)
	print len(training_data)
	b = BA_Graph(350,3)
	#print 'K MIN : {0}\n K MAX :{1}'.format(b.get_kmin(),b.get_kmax())
	t = KNN(350,3,b.average_degree(),b.get_kmin(),b.get_kmax(),b.graph.degree(90))
	t.train(training_data)
	rank = t.test()
	print b.graph.degree(90)
	
	print 'PREDICTED RANK : {0}\nACTUAL RANK :{1}'.format(rank,b.get_degreerank()[b.graph.degree(90)])
