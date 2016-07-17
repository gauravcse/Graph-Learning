import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import Queue

class BA_Graph(object) :
    def __init__(self,n,m) :
        self.graph = nx.barabasi_albert_graph(n,m)
        self.nodes = n
        self.edges = m
    def K_avg(self) :
        self.deg = sum(self.graph.degree())
        return (self.deg/self.n)
    def K_min(self) :
        return (min(self.graph.degree()))
    def K_max(self) :
        return (max(self.graph.degree()))
    
    def degree_rank(self) :
        
