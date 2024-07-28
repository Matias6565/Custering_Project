import importlib
import simpy
import functools
import random
import time
import psutil
from enum import Enum
import numpy as np
from scipy.stats import norm
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
import time
import copy
import Latency_Aware as g
import Util as util

cpu=[]

def genLogs():
	#iterate over each scheduling policy
	for i in sched_pol:
		#power consumption
		with open('Staticpower_consumption_{}.txt'.format(g.rrhs_amount),'a') as filehandle:  
		    filehandle.write("{}\n\n".format(i))
		    filehandle.writelines("%s\n" % p for p in power_consumption["{}".format(i)])
		    filehandle.write("\n")
		    #filehandle.write("\n")
		with open('Staticexecution_time_{}.txt'.format(g.rrhs_amount),'a') as filehandle:  
		    filehandle.write("{}\n\n".format(i))
		    filehandle.writelines("%s\n" % p for p in execution_time["{}".format(i)])
		    filehandle.write("\n")
		    #filehandle.write("\n")
		with open('Staticdelay_{}.txt'.format(g.rrhs_amount),'a') as filehandle:  
		    filehandle.write("{}\n\n".format(i))
		    filehandle.writelines("%s\n" % p for p in average_delay["{}".format(i)])
		    filehandle.write("\n")
		    #filehandle.write("\n")

#fog nodes amount
fog_amount = 5
#scheduling policies
sched_pol = []
sched_pol.append("removeFogFirstVPON")

#log variables
power_consumption = {}
execution_time = {}
average_delay = {}
a=0
for i in sched_pol:
	power_consumption["{}".format(i)] = []
	execution_time["{}".format(i)] = []
	average_delay["{}".format(i)] = []

def reloadDicts():
	global power_consumption, execution_time, average_delay
	for i in sched_pol:
		power_consumption["{}".format(i)] = []
		execution_time["{}".format(i)] = []
		average_delay["{}".format(i)] = []

def setExperiment(gp, rrhs, fogs):
	divided = int(rrhs/fogs)
	g.addRRHs(gp, 0, 32, "0")
	g.addRRHs(gp, 32, 64, "1")
	g.addRRHs(gp, 64, 96, "2")
	g.addRRHs(gp, 96, 128, "3")
	g.addRRHs(gp, 128, 160, "4")
	g.addRRHs(gp, 160, 192,"5")
	g.addRRHs(gp, 192, 224, "6")
	g.addRRHs(gp, 224, 256, "7")
	g.addRRHs(gp, 256, 288, "8")


#Assign VPON Heuristic simulations
#add 5 rrhs per execution
g.rrhs_amount = 10
rs = g.rrhs_amount
epochs = 12
#main loop
#for i in range(epochs):
for s in sched_pol:
	g.rrhs_amount = 5
	rs = g.rrhs_amount
	print("Executing {}".format(s))
	for i in range(epochs):
		print(g.rrhs_amount)
		importlib.reload(g)
		g.rrhs_amount = rs
		gp = g.createGraph()
		g.createRRHs()
		g.addFogNodes(gp, fog_amount)
		setExperiment(gp, g.rrhs_amount, fog_amount)
		for i in range(len(g.rrhs)):
			g.startNode(gp, "RRH{}".format(i))
			g.actives_rrhs.append("RRH{}".format(i))
		if s == "all_random":
			g.allRandomVPON(gp)
		elif s == "removeSplitVPON":
			g.cloudFirst_FogFirst(gp)
		elif s == "randomRemoveVPONs":
			g.fogFirst(gp)
		elif s == "removeFogFirstVPON":
			g.assignVPON(gp)
		#elif s == "removeSplitVPON":
		#	g.randomFogVPON(gp)
		#elif s == "most_loaded":
		#	g.assignMostLoadedVPON(gp)
		#elif s == "least_loaded":
		#	g.assignLeastLoadedVPON(gp)
		#elif s == "least_cost":
		#	g.leastCostNodeVPON(gp)
		#elif s == "least_cost_active_ratio":
		#	g.leastCostLoadedVPON(gp)
		#most and least loaded that considers the bandwidth available on each node midhaul
		#elif s == "most_loaded_bandwidth":
		#	g.assignMostLoadedVPONBand(gp)
		#elif s == "least_loaded_bandwidth":
		#	g.assignLeastLoadedVPONBand(gp)
		#elif s == "big_ratio":
		#	g.assignBigRatioVPON(gp)
		#elif s == "small_ratio":
		#	g.assignSmallRatioVPON(gp)
		start_time = g.time.time()
		mincostFlow = g.nx.max_flow_min_cost(gp, "s", "d")
		execution_time[s].append(g.time.time() - start_time)
		print(power_consumption[s].append(g.Energy2(gp)))
		#average_delay[s].append(g.overallDelay(gp))
		#print(g.rrhs_amount)
		print("Probabilidade de bloqueio {}".format(g.util.getBlockingProbability(mincostFlow)))
		if g.rrhs_amount == 100:
			print(g.fogs_vpons)
			print(g.getIncomingTraffic())
			print(g.getTotalBandwidth(gp))
			print(g.fog_activated_rrhs)
			print(gp["bridge"]["cloud"]["capacity"])


		if (a <= 12):
			cpu.append(psutil.cpu_percent())
			g.rrhs_amount += 10
			#print("Hora:"+str(a)+" cpu Atual: %", psutil.cpu_percent())
		if (a > 12):
			cpu.append(psutil.cpu_percent())
			number_of_rrhs -= incrementation
			g.rrhs_amount -= 10

		#g.rrhs_amount += 5
		rs = g.rrhs_amount
print(power_consumption)
genLogs()

'''
#Experiments for JOCN
############################# 3 RRHs per execution - 5 aggregation sites
#Assign VPON Heuristic simulations
g.rrhs_amount = 15
rs = g.rrhs_amount
#main loop
for s in sched_pol:
	g.rrhs_amount = 15
	rs = g.rrhs_amount
	print("Executing {}".format(s))
	#print(g.rrhs_amount)
	importlib.reload(g)
	g.rrhs_amount = rs
	gp = g.createGraph()
	g.createRRHs()
	g.addFogNodes(gp, fog_amount)
	setExperiment(gp, g.rrhs_amount, fog_amount)
	for i in range(len(g.rrhs)):
		g.startNode(gp, "RRH{}".format(i))
		g.actives_rrhs.append("RRH{}".format(i))
	if s == "all_random":
		g.allRandomVPON(gp)
	#elif s == "fog_first":
	#	g.fogFirst(gp)
	elif s == "cloud_fog_first":
		g.cloudFirst_FogFirst(gp)
	elif s == "cloud_first_all_fogs":
		g.assignVPON(gp)
	elif s == "cloud_first_random_fogs":
		g.randomFogVPON(gp)
	elif s == "most_loaded":
		g.assignMostLoadedVPON(gp)
	elif s == "least_loaded":
		g.assignLeastLoadedVPON(gp)
	#elif s == "least_cost":
	#	g.leastCostNodeVPON(gp)
	#elif s == "least_cost_active_ratio":
	#	g.leastCostLoadedVPON(gp)
	#most and least loaded that considers the bandwidth available on each node midhaul
	#elif s == "most_loaded_bandwidth":
	#	g.assignMostLoadedVPONBand(gp)
	#elif s == "least_loaded_bandwidth":
	#	g.assignLeastLoadedVPONBand(gp)
	#elif s == "big_ratio":
	#	g.assignBigRatioVPON(gp)
	#elif s == "small_ratio":
	#	g.assignSmallRatioVPON(gp)
	start_time = g.time.clock()
	mincostFlow = g.nx.max_flow_min_cost(gp, "s", "d")
	execution_time[s].append(g.time.clock() - start_time)
	power_consumption[s].append(g.overallPowerConsumption(gp))
	average_delay[s].append(g.overallDelay(gp))
#print(power_consumption)
genLogs("3x5")
############################# 3 RRHs per execution - 10 aggregation sites
#main loop
reloadDicts()
for s in sched_pol:
	g.rrhs_amount = 30
	rs = g.rrhs_amount
	print("Executing {}".format(s))
	#print(g.rrhs_amount)
	importlib.reload(g)
	g.rrhs_amount = rs
	gp = g.createGraph()
	g.createRRHs()
	g.addFogNodes(gp, fog_amount)
	setExperiment(gp, g.rrhs_amount, fog_amount)
	for i in range(len(g.rrhs)):
		g.startNode(gp, "RRH{}".format(i))
		g.actives_rrhs.append("RRH{}".format(i))
	if s == "all_random":
		g.allRandomVPON(gp)
	#elif s == "fog_first":
	#	g.fogFirst(gp)
	elif s == "cloud_fog_first":
		g.cloudFirst_FogFirst(gp)
	elif s == "cloud_first_all_fogs":
		g.assignVPON(gp)
	elif s == "cloud_first_random_fogs":
		g.randomFogVPON(gp)
	elif s == "most_loaded":
		g.assignMostLoadedVPON(gp)
	elif s == "least_loaded":
		g.assignLeastLoadedVPON(gp)
	#elif s == "least_cost":
	#	g.leastCostNodeVPON(gp)
	#elif s == "least_cost_active_ratio":
	#	g.leastCostLoadedVPON(gp)
	#most and least loaded that considers the bandwidth available on each node midhaul
	#elif s == "most_loaded_bandwidth":
	#	g.assignMostLoadedVPONBand(gp)
	#elif s == "least_loaded_bandwidth":
	#	g.assignLeastLoadedVPONBand(gp)
	#elif s == "big_ratio":
	#	g.assignBigRatioVPON(gp)
	#elif s == "small_ratio":
	#	g.assignSmallRatioVPON(gp)
	start_time = g.time.clock()
	mincostFlow = g.nx.max_flow_min_cost(gp, "s", "d")
	execution_time[s].append(g.time.clock() - start_time)
	power_consumption[s].append(g.overallPowerConsumption(gp))
	average_delay[s].append(g.overallDelay(gp))
#print(power_consumption)
genLogs("3x10")
'''
