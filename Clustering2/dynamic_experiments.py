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
import graph_simulator as sim
import Util
import RRH
import Latency_Aware as g
import pandas as pd
import Util
import csv

'''
from sklearn.externals import joblib
joblib.dump(clf, 'model.pkl')
import warnings

estimator.__getstate__()['_sklearn_version']

with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=UserWarning)
      estimator = joblib.load('model.pkl')
'''

#auxiliar list that keeps the plot markers and colors
markers = ['o', 'v', '^', '<', '>', 's', 'p', 'h', 'H', '+', '<','>']
colors = ['b','g','r','c','m', 'y', 'k', 'r', 'b', 'g', 'r', 'c']

#reset markers and colors
def resetMarkers():
	global markers, colors
	markers = ['o', 'v', '^', '<', '>', 's', 'p', 'h', 'H', '+', '<','>']
	colors = ['b','g','r','c','m', 'y', 'k', 'r', 'b', 'g', 'r', 'c']

#get the blocking probability from the blocked packets and the total generated packets
def calcBlocking(blocked, generated):
	blocking_probability = []
	#iterate over the collection of values in both lists
	for i in range(len(generated)):
		block_list = blocked[i]
		gen_list = generated[i]
		#now, iterate over the lists and calculates the blocking probability
		for j in range(len(gen_list)):
			if gen_list[j] == 0:
				blocking_probability.append(0.0)
			else:
				blocking_probability.append(block_list[j]/gen_list[j])
	return blocking_probability

#Logging
#generate logs
def genLogs(removeHeuristic):
	#iterate over each scheduling policy
	for i in sched_pol:
		#power consumption
		with open('dynamic_power_consumption_{}_{}_{}.txt'.format(i,removeHeuristic, g.rrhs_amount),'a') as filehandle:  
		    filehandle.write("{}\n\n".format(i))
		    filehandle.writelines("%s\n" % p for p in total_power_mean["{}".format(i)])
		    filehandle.write("\n")
		    filehandle.write("\n")
		#blocked
		with open('dynamic_blocked_{}_{}_{}.txt'.format(i,removeHeuristic, g.rrhs_amount),'a') as filehandle:  
		    filehandle.write("{}\n\n".format(i))
		    filehandle.writelines("%s\n" % p for p in total_blocking_mean["{}".format(i)])
		    filehandle.write("\n")
		    filehandle.write("\n")
	    #blocking probability
		with open('dynamic_blocking_probability_{}_{}_{}.txt'.format(i,removeHeuristic, g.rrhs_amount),'a') as filehandle:  
		    filehandle.write("{}\n\n".format(i))
		    filehandle.writelines("%s\n" % p for p in total_blocking_prob_mean["{}".format(i)])
		    filehandle.write("\n")
		    filehandle.write("\n")
		#execution times
		with open('dynamic_exec_times_{}_{}_{}.txt'.format(i,removeHeuristic, g.rrhs_amount),'a') as filehandle:  
		    filehandle.write("{}\n\n".format(i))
		    filehandle.writelines("%s\n" % p for p in total_exec_time_mean["{}".format(i)])
		    filehandle.write("\n")
		    filehandle.write("\n")

		with open('dynamic_latencia_{}_{}_{}.txt'.format(i,removeHeuristic, g.rrhs_amount),'a') as filehandle:  
		    filehandle.write("{}\n\n".format(i))
		    filehandle.writelines("%s\n" % p for p in total_latencia["{}".format(i)])
		    filehandle.write("\n")
		    filehandle.write("\n")

		with open('demands{}_{}_{}.txt'.format(i,removeHeuristic, g.rrhs_amount),'a') as filehandle:  
		    filehandle.write("{}\n\n".format(i))
		    filehandle.writelines("%s\n" % p for p in avg_demands["{}".format(i)])
		    filehandle.write("\n")
		    filehandle.write("\n")

percent_rural = []
percent_urban = []
horas=[]
#number of executions
execution_times = 2
#execution_times = 2
#scheduling policies
sched_pol = []
#sched_pol.append("EA")
#sched_pol.append("All")
#sched_pol.append("ML2")
sched_pol.append("ML")
#sched_pol.append("Split")
#vpon removing policies
remove_pol = []
#remove_pol.append("removeFogFirstVPON")
#remove_pol.append("All")
#remove_pol.append("random_remove")
#remove_pol.append("RemoveSplit")
remove_pol.append("removeSplitVPON2")

#create the lists to keep the results from
average_power = {}
average_blocking = {}
total_reqs = {}
demands={}
Split_E = []
Split_I = []
Split_D = []
Split_A = []
exec_times = {}
cobertura = []
latencia = {}
power = []
cobertura2 = {}
latencia2 = []
blocks = []
demands2 = []
blocking_prob = {}

for i in sched_pol:
	average_power["{}".format(i)] = []
	average_blocking["{}".format(i)] = []
	total_reqs["{}".format(i)] = []
	exec_times["{}".format(i)] = []
	blocking_prob["{}".format(i)] = []
	latencia["{}".format(i)] = []
	demands["{}".format(i)] = []
#resets the lists
def resetLists():
	global average_power, average_blocking, total_reqs, exec_times, demands
	#create the lists to keep the results from
	average_power = {}
	average_blocking = {}
	total_reqs = {}
	exec_times = {}
	blocking_pro = {}
	demands = {}
	latencia = {}
	for i in sched_pol:
		average_power["{}".format(i)] = []
		average_blocking["{}".format(i)] = []
		total_reqs["{}".format(i)] = []
		exec_times["{}".format(i)] = []
		blocking_prob["{}".format(i)] = []
		latencia["{}".format(i)] = []
		demands["{}".format(i)] = []


#this function reloads the graph module
def reloadGraphModule():
    importlib.reload(g)

#general function to reload modules
def reloadModule(aModule):
    importlib.reload(aModule)

blocks3 = []
power2 = []
latencia3 = []
cobertura2 = []
rural = []
urban = []
Split_E_percent = []

resetMarkers()
resetLists()

for i in sched_pol:
	print("Executions of heuristic {}".format(i))
	#begin the experiments
	for j in range(execution_times):
		print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
		print("Execution #{} of heuristic {}".format(j,i))
		print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
		#simulation environment
		env = simpy.Environment()
		#create the graph
		gp = g.createGraph()
		#create the control plane
		cp = sim.Control_Plane(env, "Graph", gp, i, "removeFogFirstVPON")
		#traffic generator
		tg = sim.Traffic_Generator(env,sim.distribution, None, cp)
		#create the rrhs
		cp.createRRHs(g.rrhs_amount,env)
		random.shuffle(g.rrhs)
		#create fog nodes
		g.addFogNodes(gp, g.fogs)
		#add RRHs to the graph
		#10 rrhs per fog node
		g.addRRHs(gp, 0, 32, "0")
		g.addRRHs(gp, 32, 64, "1")
		g.addRRHs(gp, 64, 96, "2")
		g.addRRHs(gp, 96, 128, "3")
		g.addRRHs(gp, 128, 160, "4")
		g.addRRHs(gp, 160, 192, "5")
		g.addRRHs(gp, 192, 224, "6")
		#g.addRRHs(gp, 224, 256, "7")
		#print(g.rrhs_fog)
		#starts the simulation
		env.run(until = 86401)
		average_power["{}".format(i)].append(sim.average_power_consumption)
		power.append(sim.average_power_consumption)
		power2.extend(sim.average_power_consumption)
		latencia2.append(sim.avg_latencia)
		latencia3.extend(sim.avg_latencia)
		blocks.append(sim.average_blocking_prob)
		blocks3.extend(sim.average_blocking_prob)
		#demands.extend(sim.total_demands)
		demands["{}".format(i)].append(sim.total_demands)
		#print(demands)
		demands2.extend(sim.total_demands)
		#print(sim.average_power_consumption)
		average_blocking["{}".format(i)].append(sim.average_blocking_prob)
		total_reqs["{}".format(i)].append(sim.total_requested)
		#print("Service Availability {}".format(sim.avg_service_availability))
		#cobertura2["{}".format(i)].append(sim.avg_service_availability)
		cobertura.append(sim.avg_service_availability)
		cobertura2.extend(sim.avg_service_availability)
		rural.extend(sim.rural)
		urban.extend(sim.urban)
		Split_E.extend(sim.Split_E)
		Split_I.extend(sim.Split_I)
		Split_D.extend(sim.Split_D)
		Split_A.extend(sim.Split_A)
		horas.extend(sim.horas)
		Split_E_percent.extend(sim.Split_Tot)
		percent_urban.extend(sim.percent_urban)
		percent_rural.extend(sim.percent_rural)
		#print(cobertura)
		#print("Atraso{}".format(atraso_fila(gp)))
		exec_times["{}".format(i)].append(sim.average_execution_time)
		latencia["{}".format(i)].append(sim.avg_latencia)
		blocking_prob["{}".format(i)].append(calcBlocking(average_blocking["{}".format(i)], total_reqs["{}".format(i)]))
		#print("Mudança do arrival rate {}".format(sim.change_load.arrival_rate))
		reloadGraphModule()
		reloadModule(sim)

np.random.seed(1)
#dados = pd.DataFrame(data={"Energia": power2, "Block": blocks3, "Latencia":latencia3, "Cobertura":cobertura2, "Demands":demands2,"Rural":rural, "Urban":urban})
#dados = pd.DataFrame(data={"Rural":rural, "Urban":urban, "Percent_Urban":percent_urban, "Percent_Rural":percent_rural})
#dados = pd.DataFrame(data={"Percent_Urban":percent_urban, "Percent_Rural":percent_rural})
dados = pd.DataFrame(data={"Split_E":Split_E,"Split_I":Split_I,"Split_D":Split_D,"Split_A":Split_A, "Split_tots":Split_E_percent})
#dados = pd.DataFrame(data={"Split_E":Split_E})
dados.to_csv("Splits.csv", sep=';',index=False)

np.random.seed(1)
dados2 = pd.DataFrame(data={"Energia": power2, "Block": blocks3, "Latencia":latencia3, "Cobertura":cobertura2, "Demands":demands2, "Horas":horas})
#dados = pd.DataFrame(data={"Rural":rural, "Urban":urban, "Percent_Urban":percent_urban, "Percent_Rural":percent_rural})
#dados = pd.DataFrame(data={"Percent_Urban":percent_urban, "Percent_Rural":percent_rural})
#dados = pd.DataFrame(data={"Split_E":Split_E,"Split_I":Split_I,"Split_D":Split_D,"Split_A":Split_A, "Split_tots":Split_E_percent})
#dados = pd.DataFrame(data={"Split_E":Split_E})
dados2.to_csv("Metrics.csv", sep=';',index=False)


np.random.seed(1)
dados3 = pd.DataFrame(data={"Rural":rural, "Urban":urban})
#dados = pd.DataFrame(data={"Rural":rural, "Urban":urban, "Percent_Urban":percent_urban, "Percent_Rural":percent_rural})
#dados = pd.DataFrame(data={"Percent_Urban":percent_urban, "Percent_Rural":percent_rural})
#dados = pd.DataFrame(data={"Split_E":Split_E,"Split_I":Split_I,"Split_D":Split_D,"Split_A":Split_A, "Split_tots":Split_E_percent})
#dados = pd.DataFrame(data={"Split_E":Split_E})
dados3.to_csv("City.csv", sep=';',index=False)



#calculate the means from the executions
#power consumption means

#blocking means
total_blocking_mean = {}
for i in sched_pol:
	total_blocking_mean["{}".format(i)] = [float(sum(col))/len(col) for col in zip(*average_blocking["{}".format(i)])]

#amount of demands
avg_total_demands = {}
for i in sched_pol:
	avg_total_demands["{}".format(i)] = [float(sum(col))/len(col) for col in zip(*demands["{}".format(i)])]


#execution times means
total_exec_time_mean = {}
for i in sched_pol:
	total_exec_time_mean["{}".format(i)] = [float(sum(col))/len(col) for col in zip(*exec_times["{}".format(i)])]

#blocking probability means
total_blocking_prob_mean = {}
for i in sched_pol:
	total_blocking_prob_mean["{}".format(i)] = [float(sum(col))/len(col) for col in zip(*blocking_prob["{}".format(i)])]

#blocking means
total_latencia = {}
for i in sched_pol:
	total_latencia["{}".format(i)] = [float(sum(col))/len(col) for col in zip(*latencia["{}".format(i)])]

total_power_mean = {}
for i in sched_pol:
	total_power_mean["{}".format(i)] = [float(sum(col))/len(col) for col in zip(*average_power["{}".format(i)])]

avg_demands = {}
for i in sched_pol:
	avg_demands["{}".format(i)] = [float(sum(col))/len(col) for col in zip(*demands["{}".format(i)])]

avg_cobertura= []
avg_cobertura = [float(sum(col))/len(col) for col in zip(*cobertura)]

#avg_demands= []
#avg_demands = [float(sum(col))/len(col) for col in zip(*demands)]


power2_mean = []
power2_mean = [float(sum(col))/len(col) for col in zip(*power)]

genLogs("remove_cloud_first")

with open('Results/avg_cobertura.txt','w') as filehandle:  
    filehandle.write("Split\n\n")
    filehandle.writelines("%s\n" % p for p in avg_cobertura)
    filehandle.write("\n")

with open('Results/demands2.txt','w') as filehandle:  
    filehandle.write("Split\n\n")
    filehandle.writelines("%s\n" % p for p in demands2)
    filehandle.write("\n")

with open('Results/demands.txt','w') as filehandle:  
    filehandle.write("Split\n\n")
    filehandle.writelines("%s\n" % p for p in avg_total_demands)
    filehandle.write("\n")



def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    #return  h, m-h, m+h
    return h

#confidence intervals
power_ci = []
power_ci = [mean_confidence_interval(col, confidence = 0.95) for col in zip(*power)]

latencia_ci = []
latencia_ci = [mean_confidence_interval(col, confidence = 0.95) for col in zip(*latencia2)]

bloqueio_ci = []
bloqueio_ci = [mean_confidence_interval(col, confidence = 0.95) for col in zip(*blocks)]

cobertura_ci = []
cobertura_ci = [mean_confidence_interval(col, confidence = 0.95) for col in zip(*cobertura)]

with open('Results/power_ci.txt','w') as filehandle:  
    filehandle.write("Ci\n\n")
    filehandle.writelines("%s\n" % p for p in power_ci)
    filehandle.write("\n")

with open('Results/latencia_ci.txt','w') as filehandle:  
    filehandle.write("Ci\n\n")
    filehandle.writelines("%s\n" % p for p in latencia_ci)
    filehandle.write("\n")

with open('Results/bloqueio_ci.txt','w') as filehandle:  
    filehandle.write("Ci\n\n")
    filehandle.writelines("%s\n" % p for p in bloqueio_ci)
    filehandle.write("\n")

with open('Results/cobertura_ci.txt','w') as filehandle:  
    filehandle.write("Ci\n\n")
    filehandle.writelines("%s\n" % p for p in cobertura_ci)
    filehandle.write("\n")


#plot results
#generate the plots for power consumption
for i in sched_pol:
	plt.plot(total_power_mean["{}".format(i)], color = '{}'.format(colors.pop()),marker='{}'.format(markers.pop()), label = "{}".format(i))
plt.ylabel('Power Consumption')
plt.xlabel("Day Time")
plt.legend(loc="upper left",prop={'size': 6})
#plt.grid()
plt.savefig('dynamic_power{}_remove_cf.png'.format(g.rrhs_amount), bbox_inches='tight')
#plt.show()
plt.clf()

resetMarkers()

#generate the plots for blocking
for i in sched_pol:
	plt.plot(total_blocking_mean["{}".format(i)], color = '{}'.format(colors.pop()), marker='{}'.format(markers.pop()), label = "{}".format(i))
plt.ylabel('Blocking Amount')
plt.xlabel("Day Time")
plt.legend(loc="upper left",prop={'size': 6})
#plt.grid()
plt.savefig('dynamic_blocking{}_remove_cf.png'.format(g.rrhs_amount), bbox_inches='tight')
#plt.show()
plt.clf()

resetMarkers()
#generate the plots for execution times
for i in sched_pol:
	plt.plot(total_exec_time_mean["{}".format(i)], color = '{}'.format(colors.pop()), marker='{}'.format(markers.pop()), label = "{}".format(i))
plt.ylabel('Execution Time')
plt.xlabel("Day Time")
plt.legend(loc="upper left",prop={'size': 6})
#plt.grid()
plt.savefig('dynamic_execution_time{}__remove_cf.png'.format(g.rrhs_amount), bbox_inches='tight')
#plt.show()
plt.clf()

resetMarkers()
#generate the plots for blocking probability
for i in sched_pol:
	plt.plot(total_blocking_prob_mean["{}".format(i)], color = '{}'.format(colors.pop()), marker='{}'.format(markers.pop()), label = "{}".format(i))
plt.ylabel('Blocking Probability')
plt.xlabel("Day Time")
plt.legend(loc="upper left",prop={'size': 6})
#plt.grid()
plt.savefig('dynamic_blocking_probability{}_remove_cf.png'.format(g.rrhs_amount), bbox_inches='tight')
#plt.show()
plt.clf()


resetMarkers()
#generate the plots for blocking probability
for i in sched_pol:
	plt.plot(total_latencia["{}".format(i)], color = '{}'.format(colors.pop()), marker='{}'.format(markers.pop()), label = "{}".format(i))
plt.ylabel('Latency')
plt.xlabel("Day Time")
plt.legend(loc="upper left",prop={'size': 6})
#plt.grid()
plt.savefig('dynamic_latency{}_remove_cf.png'.format(g.rrhs_amount), bbox_inches='tight')
#plt.show()
plt.clf()


