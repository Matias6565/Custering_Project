import importlib
import simpy
import functools
import random
import time
from enum import Enum
import numpy as np
from scipy.stats import norm
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
import copy
import Util
import RRH
#import Latency_Aware as g
import Latency_Aware as g
import pandas as pd
import Util
import csv


def genLogs():
	#iterate over each scheduling policy
	for i in sched_pol:
		#power consumption
		with open('/home/matias/Dropbox/Graph-based Heuristic/split_graph/Data/Static/power_consumption_{}.txt'.format(g.rrhs_amount),'a') as filehandle:  
		    filehandle.write("{}\n\n".format(i))
		    filehandle.writelines("%s\n" % p for p in power_consumption["{}".format(i)])
		    filehandle.write("\n")
		    #filehandle.write("\n")
		with open('/home/matias/Dropbox/Graph-based Heuristic/split_graph/Data/Static/execution_time_{}.txt'.format(g.rrhs_amount),'a') as filehandle:  
		    filehandle.write("{}\n\n".format(i))
		    filehandle.writelines("%s\n" % p for p in execution_time["{}".format(i)])
		    filehandle.write("\n")
		    #filehandle.write("\n")
		with open('/home/matias/Dropbox/Graph-based Heuristic/split_graph/Data/Static/delay_{}.txt'.format(g.rrhs_amount),'a') as filehandle:  
		    filehandle.write("{}\n\n".format(i))
		    filehandle.writelines("%s\n" % p for p in average_delay["{}".format(i)])
		    filehandle.write("\n")
		    #filehandle.write("\n")
		with open('/home/matias/Dropbox/Graph-based Heuristic/split_graph/Data/Static/bloqueio{}.txt'.format(g.rrhs_amount),'a') as filehandle:  
		    filehandle.write("{}\n\n".format(i))
		    filehandle.writelines("%s\n" % p for p in bloqueio["{}".format(i)])
		    filehandle.write("\n")
		    #filehandle.write("\n")
		#with open('/home/matias/Dropbox/Graph-based Heuristic/split_graph/Data/Static/lambdas{}.txt'.format(g.rrhs_amount),'a') as filehandle:  
		    #filehandle.write("{}\n\n".format(i))
		    #filehandle.writelines("%s\n" % p for p in lambdas_necessary["{}".format(i)])
		    #filehandle.write("\n")
		    #filehandle.write("\n")

#lambdas_necessary


#overallPowerConsumption
#fog nodes amount
fog_amount = 3
#scheduling policies
sched_pol = []
sched_pol.append("cloud_fog_first")
power_consumption = {}
execution_time = {}
antenas = {}
average_delay = {}
bloqueio = {}
lamb_necessary = []
fog_lamb_necessary = []
rrhs = []

for i in sched_pol:
	power_consumption["{}".format(i)] = []
	execution_time["{}".format(i)] = []
	average_delay["{}".format(i)] = []
	bloqueio["{}".format(i)] = []
	#lamb_necessary["{}".format(i)] = []

def setExperiment(gp, rrhs, fogs):
	divided = int(rrhs/fogs)
	g.addRRHs(gp, 0, divided, "0")
	g.addRRHs(gp, divided, divided*2, "1")
	g.addRRHs(gp, divided*2, divided*3, "2")
	g.addRRHs(gp, divided*3, divided*4, "3")
	g.addRRHs(gp, divided*4, divided*5, "4")
	#g.addRRHs(gp, divided*5, divided*6, "5")
	#g.addRRHs(gp, divided*6, divided*7, "6")
	#g.addRRHs(gp, divided*7, divided*8, "7")

#add 5 rrhs per execution
#g.rrhs_amount = 5
rs = g.rrhs_amount
#main loop
#for s in sched_pol:
g.rrhs_amount = 5
rs = g.rrhs_amount
lambdas_necessary = []
#print("Executing {}".format(s))
for i in range(28):
	#print(g.rrhs_amount)
	importlib.reload(g)
	g.rrhs_amount = rs
	rrhs.append(g.rrhs_amount)
	gp = g.createGraph()
	g.createRRHs()
	g.addFogNodes(gp, fog_amount)
	setExperiment(gp, g.rrhs_amount, fog_amount)
	for i in range(len(g.rrhs)):
		g.startNode(gp, "RRH{}".format(i))
		g.actives_rrhs.append("RRH{}".format(i))
	s ="cloud_fog_first"
	g.Latency_aware(gp)
	#g.CFRAN_Badwidth_aware(gp)
	start_time = g.time.process_time()
	mincostFlow = g.nx.max_flow_min_cost(gp, "s", "d")
	execution_time[s].append(g.time.process_time() - start_time)
	power_consumption[s].append(Util.overallPowerConsumption(gp))
	#average_delay[s].append(Util.overallDelay(gp))
	bloqueio[s].append(Util.getBlockingProbability(gp))
	lamb_necessary.append(Util.get_CloudLambdasNecessary(gp))
	fog_lamb_necessary.append(Util.get_Fog_LambdasNecessary(gp,g.rrhs_amount))
	#print("LambdasFog {}".format(Util.get_Fog_LambdasNecessary(gp,g.rrhs_amount)))
	#print("Lambdas Necessários {}".format(Util.get_CloudLambdasNecessary(gp)))
	#print("Lambdas Necessários {}".format(bloqueio[s]))
	#atraso.append(Util.atraso_fila(gp))
	#print("Atraso{}".format(Util.atraso_fila(gp)))
	#print("Amount of activated RRHS {}: ".format(g.rrhs_amount))
	#print(Util.getBlockingProbability(mincostFlow))
	g.rrhs_amount += 5
	rs = g.rrhs_amount
print(power_consumption)
print(execution_time)
genLogs()


np.random.seed(1)
dados = pd.DataFrame(data={"RRHs":rrhs, "Coud_Lambdas": lamb_necessary, "Fog_Labdas" : fog_lamb_necessary})
dados.to_csv("/home/matias/Dropbox/Graph-based Heuristic/split_graph/Data/Static/JOCN.csv", sep=';',index=False)

