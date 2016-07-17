import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import Queue

training_data = list()

def Train(graph) :
	training_data.append(graph)

class KNN(object) :
	def __init__(self,n,m,k,K = 10) :
		self.nodes = n
		self.edges = m
		self.average_degree = k
		self.neighbours = K
		self.q = Queue.PriorityQueue(self.neighbours)
		self.kmin,self.kmax = 0.0,0.0
		
	def compute_distance(self,g) :
		return sqrt(((g.average_degree - self.average_degree)**2) + (g.edge - self.edges)**2) + ((g.nodes - self.nodes)**2))

	def algorithm(self,data) :
		self.training_data = data
		for graph in self.training_data :
			euclidian_distance = self.compute_distance(graph)
			if self.q.full() == False :
				tup = (-1*euclidian_distance,graph)
				self.q.put(tup)
			else if self.q.full() == True :
				tup1 = self.q.get()
				tup2  = (-1*euclidian_distance,graph)
				if (tup1[0] < tup2[0]) :
					self.q.put(tup2)
				else if (tup1[0] >= tup2[0]) :
					self.q.put(tup1)
		for i in xrange(self.neighbours) :
			tup = self.q.get()
			graph = tup[1]
			self.kmin += graph.get_kmin()
			self.kmax += graph.get_kmax()
		
			if self.q.full() == True :
				break
		self.kmin = float(self.kmin) / float(self.neighbours)
		self.kmax = float(self.kmax) / float(self.neighbours)
		
	def rank(self,degree = 0,gamma = 2.5) :
		self.A = float(n)/(pow(self.kmin,(1-gamma)) - pow(self.max,(1-gamma)))
		self.B = (float(n)*float(pow(self.kmax,(1-gamma))))/(pow(self.kmin,(1-gamma)) - pow(self.max,(1-gamma)))
		if degree == 0 :
			self.degree_rank = dict()
			iterator = 1
			while True :
				self.X = pow(iterator,(1-gamma))
				self.degree_rank[iterator] = A*X - B
				if self.degree_rank[iterator] < 1 :
					break				
				iterator += 1
			return self.degree_rank
		else :
			self.X = pow(degree,(1-gamma))
			self.predicted_rank = A*X - B
			return self.predicted_rank
		
