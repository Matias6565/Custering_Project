#import networkx as nx
#import matplotlib.pyplot as plt
#import math

class RRH(object):
    def __init__(self, ecpri, rrhId):
        self.ecpri = ecpri
        self.id = "RRH{}".format(rrhId)

