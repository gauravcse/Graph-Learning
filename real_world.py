'''
    created by Gaurav Mitra

    INSTRUCTIONS :  the input training files has to be 70 in number and the filename should be 'file*.pickle'
                    where * ranges from 1 to 70.
                    example :   file1.pickle
                                file2.pickle
                                upto file70.pickle
                    the test data starts from 71 and goes till 100
                    example :   file71.pickle
                                file72.pickle
                                file100.pickle

                    output :    output.txt
'''
try :
    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx
    import Queue
    from math import sqrt
    import random
except ImportError :
    print "Import Statement Error"
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
            self.rank = g.get_degreerank()		# DICT : degree : rankdegree = self.rank.keys()
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
    def __init__(self,filename) :
        #self.graph = nx.barabasi_albert_graph(100,5)
        self.graph = nx.Graph()
        self.graph = nx.read_gpickle(filename)
        self.nodes = self.graph.nodes()
        self.graph = self.mapper()
        self.edges = self.graph.edges()
    def mapper(self) :
        self.mapping = dict()
        for i in xrange(len(self.nodes)) :
            self.mapping[G.nodes()[i]] = i
        self.graph = nx.relabel_nodes(self.graph,self.mapping)
        return self.graph
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
    f = open('output.txt','a')
    filename = "file.txt"
    for i in xrange(70) :
        name = filename.split(".")[0]
        extension = filename.split(".")[1]
        fname = name + str(i+1) + "." + extension
        b = BA_Graph(fname)
        Train(b)
        print 'Train = {}'.format(i+1)
    for i in xrange(30) :
        name = filename.split(".")[0]
        extension = filename.split(".")[1]
        fname = name + str(71+i) + "." + extension
        b = BA_Graph(fname)
        count = 0
        while count < 10000 :
            node_for_degree = random.randint(0,b.get_nodes())
            t = KNN(b.get_nodes(),b.get_edges(),b.average_degree(),b.get_kmax(),b.graph.degree(node_for_degree),30)
            t.train(training_data)
            rank = t.test()
            f.write(str(rank)+","+str(b.get_degreerank()[b.graph.degree(node_for_degree)])+"\n")
            count += 1
        print 'Test = {}'.format(i+1)
    f.close()
