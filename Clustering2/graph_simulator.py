import simpy
import functools
import random as np
import time
from enum import Enum
import numpy
from scipy.stats import norm
import matplotlib.pyplot as plt
import copy
import sys
import Util
import RRH
import Latency_Aware as g
import pandas as pd
import Util
import networkx as nx



#keeps the transmission delays
urban = []
rural = []
percent_urban = []
percent_rural = []
Split_E_percent = []
horas = []


delay_time = []
average_delay_time = []
#keeps the execution time
execution_time = []
average_execution_time = []
#keeps the blocking probability
blocking_prob = 0
average_blocking_prob = []
#keeps the power consumption
power_consumption = []
average_power_consumption = []
latencia = []
Split_Tot = []
Split_E =[]
Split_I =[]
Split_D =[]
Split_A =[]
avg_latencia = []
avg_service_availability = []
avg_total_allocated = []
#keeps the average delay on each node
proc_nodes_delay = []
average_proc_nodes_delay = []
#cpri line rate
ecpri = 1966
#count the total of requested RRHs
total_requested = []
#count allocated requests
sucs_reqs = 0
total_allocated = []
#network_threshold = 0.8
#traffic_quocient = 130 #cenário split fs
traffic_quocient = 80
#traffic_quocient = 50
rrhs_quantity = 80
served_requests = 0
#timestamp to change the load
change_time = 3600
#the next time
next_time = 3600
#the actual hout time stamp
actual_stamp = 0.0
#inter arrival rate of the users requests
arrival_rate = 3600
#service time of a request
service_time = lambda x: np.uniform(0,100)
#total generated requests per timestamp
total_period_requests = 0
#to generate the traffic load of each timestamp
loads = []
actives = []
#number of timestamps of load changing
stamps = 24
hours_range = range(1, stamps+1)
for i in range(stamps):
	x = norm.pdf(i, 12, 3)
	x *= traffic_quocient
	#x= round(x,4)
	#if x != 0:
	#	loads.append(x)
	loads.append(x)
#distribution for arrival of packets
#first arrival rate of the simulation - to initiate the simulation
arrival_rate = loads[0]/change_time
distribution = lambda x: np.expovariate(arrival_rate)
loads.reverse()
#print(loads)
stamps = len(loads)
#record the requests arrived at each stamp
traffics = []
total_demands = []
#amount of rrhs
rrhs_amount = 80
#list of rrhs of the network
rrhs = []
service_rate = 0.00001

#traffic generator - generates requests considering the distribution
class Traffic_Generator(object):
	def __init__(self, env, distribution, service, cp):
		self.env = env
		self.dist = distribution
		self.service = service
		self.cp = cp
		self.req_count = 0
		self.action = self.env.process(self.run())
		self.load_variation = self.env.process(self.change_load())

	#generation of requests
	def run(self):
		global total_period_requests
		global rrhs
		#global actives
		while True:
			#if rrhs:
			#if total_period_requests <= maximum_load:
			yield self.env.timeout(self.dist(self))
			#total_period_requests +=1
			self.req_count += 1
			#takes the first turned off RRH
			if g.rrhs:
				r = g.rrhs.pop()
				self.cp.requests.put(r)
				#print("Took {}".format(r.id))
				#r.updateGenTime(self.env.now)
				#r.enabled = True
				total_period_requests +=1
				#np.shuffle(rrhs)
			else:
				#print("Empty at {}".format(self.env.now))
				pass


	#changing of load
	def change_load(self):
		while True:
			global traffics
			#global loads
			global arrival_rate
			global total_period_requests
			global next_time
			global power_consumption
			global incremental_power_consumption
			global batch_power_consumption
			global inc_batch_power_consumption
			global incremental_blocking
			global batch_blocking
			global served_requests
			global sucs_reqs
			#self.action = self.action = self.env.process(self.run())
			yield self.env.timeout(change_time)
			actual_stamp = self.env.now
			#print("next time {}".format(next_time))
			next_time = actual_stamp + change_time
			traffics.append(total_period_requests)
			arrival_rate = loads.pop()/change_time
			if total_period_requests >0:
				total_wait = 0
				queue_size = 0
				mu = 1.0/service_rate
				#mu = 1.0/service_time
				l = 1.0/arrival_rate
				rho = l/mu
				W = rho/mu/(1-rho)  # average weight in the queue
				T = 1/mu/(1-rho)    # average total system time.
				nq_bar = rho/(1.0 - rho) - rho # The average number waiting in the queue
				#if W and T and nq_bar >0:
					#print (" Avg. Espera na Fila: {:.15f}\n Avg. Tempo total: {:.15f}\n Avg Tamanho da Fila {:.15f}\n".format(W, T, nq_bar))
				#else:
					#print ("Avg. Espera na Fila: 0 \n Avg. Tempo total: 0 \n Avg Tamanho da Fila 0\n")
				#print ('Sim Average queue wait = {}'.format(queue_wait/total_period_requests))
				#print ('Sim Average total wait = {}'.format(total_wait/total_period_requests))
				#print ('Sim Average queue size = {}'.format(queue_size/float(total_period_requests)))
				#print ("Theory: avg queue wait {:.10f}, avg total time {:.10f}, avg queue size {:.10f}".format(W, T, nq_bar))
				#print ('Avg. Espera na Fila = {}'.format(queue_wait/total_period_requests))
				#print ('Avg. Espera Total = {}'.format(total_wait/total_period_requests))
				#print ('Avg. Tam. Fila = {}'.format(queue_size/float(total_period_requests)))

				#service_rate = random.expovariate(1 / total_period_requests)
				#print(1/(service_rate - (1-(arrival_rate/service_rate))))# veja isso direito depois
			else:
				print("0")
			self.action = self.env.process(self.run())
			self.cp.countAverages()
			print("Arrival rate now is {} at {} and was generated {}".format(arrival_rate, self.env.now/3600, total_period_requests))
			horas.append(self.env.now/3600)
			total_requested.append(total_period_requests)
			total_period_requests = 0
			sucs_reqs = 0


#control plane that controls the allocations and deallocations
class Control_Plane(object):
	def __init__(self, env, type, graph, vpon_scheduling, vpon_remove):
		self.env = env
		self.requests = simpy.Store(self.env)
		self.departs = simpy.Store(self.env)
		self.action = self.env.process(self.run())
		self.deallocation = self.env.process(self.depart_request())
		self.type = type
		self.graph = graph
		self.vpon_scheduling = vpon_scheduling
		self.vpon_remove = vpon_remove
		
	#account the metrics of the simulation
	def countAverages(self):
		global served_requests, total_period_requests
		global blocking_prob, power_consumption, execution_time
		
		if blocking_prob != 0:
			average_blocking_prob.append(blocking_prob)
		else:
			average_blocking_prob.append(0)
		blocking_prob = 0
		if power_consumption:
			average_power_consumption.append(numpy.mean(power_consumption))
		else:
			average_power_consumption.append(0)
		#power_consumption = []
		if execution_time:
			average_execution_time.append(numpy.mean(execution_time))
		else:
			average_execution_time.append(0)
		execution_time = []
		if total_period_requests>0:
			if served_requests/total_period_requests > 1:
				avg_service_availability.append(1)
			else:
				avg_service_availability.append(served_requests/total_period_requests)
			avg_total_allocated.append(served_requests)
			#served_requests = 0
		else:
			avg_service_availability.append(0)
			avg_total_allocated.append(0)
		if latencia:
			avg_latencia.append(numpy.mean(latencia))
		else:
			avg_latencia.append(0)

		if total_period_requests>0:
			total_demands.append(total_period_requests)
		else:
			total_demands.append(total_period_requests)



		if total_period_requests:
			for i in range(total_period_requests):
				if Util.distancia(i)>40:
					rural.append(1)
				else:
					rural.append(0)

		else:
			rural.append(0)

		if total_period_requests:
			for i in range(total_period_requests):
				#print(i,Util.distancia(i))
				if Util.distancia(i)<=40:
					urban.append(1)
				else:	
					urban.append(0)

		else:
			urban.append(0)


		
		if urban:
			percent_urban.append(numpy.mean(urban))
		else:
			percent_urban.append(0)

		if rural:
			percent_rural.append(numpy.mean(rural))
		else:
			percent_rural.append(0)

		if Split_Tot:
			Split_Tot.append(numpy.mean(Split_Tot))
			#Split_E_percent.append(Split_Tot)
		#if Split_E:
		#	Split_E.append(numpy.mean(Split_E))

		if Split_E:
			Split_E.append(numpy.mean(Split_E))

		if Split_I:
			Split_I.append(numpy.mean(Split_I))
		if Split_D:
			Split_D.append(numpy.mean(Split_D))
		if Split_A:
			Split_A.append(numpy.mean(Split_A))


	#create rrhs
	def createRRHs(self, amount, env):
		for i in range(amount):
   			g.rrhs.append(RRH(ecpri, i, self, self.env))


	#take requests and tries to allocate on a RRH
	def run(self):
		global blocking_prob
		while True:
			r = yield self.requests.get()
			#print("Got {}".format(r.id))
			#turn the RRH on
			g.startNode(self.graph, r.id)
			g.actives_rrhs.append(r.id)
			g.addActivated(r.id)
			#calls the allocation of VPONs
			#all random vpon scheduling
			if self.vpon_scheduling == "EA":
				g.CFRAN_Badwidth_aware(self.graph)
			if self.vpon_scheduling == "All":
				g.FS_Split_Badwidth_aware(self.graph)
			if self.vpon_scheduling == "Split":
				g.Latency_aware(self.graph)
			if self.vpon_scheduling == "ML":
				g.Clustering_Split_Latency_aware(self.graph)
			if self.vpon_scheduling == "ML2":
				g.Clustering_Split_Latency_aware2(self.graph)				
			#execute the max cost min flow heuristic
			start_time = time.process_time()
			mincostFlow = g.nx.max_flow_min_cost(self.graph, "s", "d")
			running_time = time.process_time() - start_time
			if g.getProcessingNodes(self.graph, mincostFlow, r.id):
				self.env.process(r.run())
				power_consumption.append(Util.overallPowerConsumption(self.graph))
				Split_E.append(g.Split_E)
				Split_I.append(g.Split_I)
				Split_D.append(g.Split_D)
				Split_A.append(g.Split_A)
				Split_Tot.append(g.Split_Tot)
				latencia.append(Util.Total_Queue_Delay_dinamico(r.id,self.graph))#pega a fila
				#latencia.append(Util.Total_Delay(r.id,self.graph))
				#print(Util.overallPowerConsumption(self.graph))
				#print("Energia {}".format(power_consumption))
				execution_time.append(running_time)
				#g.addActivated(r.id)
				#print("++++++++++++++++++++++++++++++")
				#print("Fogs ativas {}".format(g.fog_activated_rrhs))
				#print("Foi {}".format(g.activatedFogRRHs()))
				#print("++++++++++++++++++++++++++++++")
				#g.getProcessingNodes(self.graph, mincostFlow, r.id)#USAR ESSE METODO PARA VER SE FOI POSSÍVEL COLOCAR O FLUXO DO RRH EM ALGUM NÓ (MODIFICAR A GETpROCESSING PRA RETORNAR O NÓ QUE ELE FOI POSTO)
				#print("Inserted {}".format(r.id))
				#print(mincostFlow[r.id])
			else:
				blocking_prob += 1
				print("No flow was found!")
				g.minusActivated(r.id)
				g.endNode(self.graph, r.id)
				g.actives_rrhs.remove(r.id)
				g.rrhs.append(r)
				np.shuffle(g.rrhs)
				power_consumption.append(Util.overallPowerConsumption(self.graph))
				latencia.append(Util.Total_Queue_Delay_dinamico(r.id,self.graph))#pega a fila
				#print("Energia {}".format(power_consumption))
	
	#starts the deallocation of a request
	def depart_request(self):
		global served_requests
		while True:
			served_requests += 1
			r = yield self.departs.get()
			#print("Departing {}".format(r.id))
			g.actives_rrhs.remove(r.id)
			#print("Removing {} from {}".format(r.id, g.rrhs_proc_node[r.id]))
			g.removeRRHNode(r.id)
			g.minusActivated(r.id)
			g.rrhs.append(r)
			g.endNode(self.graph, r.id)
			np.shuffle(g.rrhs)
			#choose the heuristic to remove VPONs
			if self.vpon_remove == "fog_first":
				g.removeVPON(self.graph)
			if self.vpon_remove == "randon_remove":
				g.randomRemoveVPONs(self.graph)
			if self.vpon_remove == "RemoveSplit":
				g.removeSplitVPON(self.graph)
			if self.vpon_remove == "removeSplitVPON2":
				g.removeSplitVPON2(self.graph)
			#if self.vpon_remove == "random_remove":
				#g.randomRemoveVPONs(self.graph)
			power_consumption.append(Util.overallPowerConsumption(self.graph))
			#print("Departed Request")
			#print("Cloud is {}".format(g.cloud_vpons))
	
	#to capture the state of the network at a given rate - will be used to take the metrics at a given (constant) moment
	def checkNetwork(self):
		while True:
			yield self.env.timeout(1800)
			print("Taking network status at {}".format(self.env.now))
			print("Total generated requests is {}".format(total_period_requests))

#rrh
class RRH(object):
	def __init__(self, ecpri, rrhId, cp, env):
		self.ecpri = ecpri
		self.id = "RRH{}".format(rrhId)
		self.cp = cp
		self.env = env

	def run(self):
		t = np.uniform((next_time -self.env.now)/4, next_time -self.env.now)
		yield self.env.timeout(t)
		self.cp.departs.put(self)



	#reset the parameters
	def resetParams(self):
		global count, change_time, next_time, actual_stamp, arrival_rate, service_time, total_period_requests, loads, actives, stamps, hours_range, arrival_rate, distribution,traffics, total_demands, percent_rural, percent_urban, rural, urban, Split_Tot
		global power_consumption,average_power_consumption,	batch_power_consumption,batch_average_consumption,incremental_blocking,batch_blocking
		global redirected,activated_nodes,average_act_nodes,b_activated_nodes,b_average_act_nodes, latencia, avg_latencia, Split_E,Split_I,Split_D,Split_A
		global activated_lambdas,average_act_lambdas,b_activated_lambdas,b_average_act_lambdas,	activated_dus,average_act_dus,b_activated_dus, Split_E_percent, horas
		global b_average_act_dus,activated_switchs,	average_act_switch,	b_activated_switchs,b_average_act_switch,redirected_rrhs,average_redir_rrhs
		global b_redirected_rrhs,b_average_redir_rrhs,time_inc,	avg_time_inc,time_b,avg_time_b,count_cloud,	count_fog,b_count_cloud,b_count_fog
		global max_count_cloud,	average_count_fog,b_max_count_cloud,b_average_count_fog,batch_rrhs_wait_time,avg_batch_rrhs_wait_time
		global inc_batch_count_cloud, inc_batch_max_count_cloud, inc_batch_count_fog, inc_batch_average_count_fog, time_inc_batch, avg_time_inc_batch
		global inc_batch_redirected_rrhs, inc_batch_average_redir_rrhs, inc_batch_power_consumption, inc_batch_average_consumption, inc_batch_activated_nodes
		global inc_batch_average_act_nodes, inc_batch_activated_lambdas, inc_batch_average_act_lambdas,	inc_batch_activated_dus, inc_batch_average_act_dus
		global inc_batch_activated_switchs, inc_batch_average_act_switch
		global inc_blocking, total_inc_blocking, batch_blocking, total_batch_blocking, inc_batch_blocking, total_inc_batch_blocking
		global external_migrations, internal_migrations, avg_external_migrations, avg_internal_migrations, served_requests
		global lambda_usage, avg_lambda_usage,proc_usage, avg_proc_usage
		global act_cloud, act_fog, avg_act_cloud, avg_act_fog, daily_migrations
		global count_ext_migrations, total_service_availability, avg_service_availability, avg_total_allocated, total_requested

		total_requested = []
		served_requests = 0
		count = 0
		#timestamp to change the load
		change_time = 3600
		#the next time
		next_time = 3600
		#the actual hout time stamp
		actual_stamp = 0.0
		#inter arrival rate of the users requests
		arrival_rate = 3600
		#service time of a request
		service_time = lambda x: np.uniform(0,100)
		#total generated requests per timestamp
		total_period_requests = 0
		#to generate the traffic load of each timestamp
		loads = []
		actives = []
		#number of timestamps of load changing
		stamps = 24
		hours_range = range(1, stamps+1)
		for i in range(stamps):
			x = norm.pdf(i, 12, 3)
			x *= traffic_quocient
			#x= round(x,4)
			#if x != 0:
			#	loads.append(x)
			loads.append(x)
		#distribution for arrival of packets
		#first arrival rate of the simulation - to initiate the simulation
		arrival_rate = loads[0]/change_time
		distribution = lambda x: np.expovariate(arrival_rate)
		loads.reverse()
		#print(loads)
		stamps = len(loads)
		#record the requests arrived at each stamp
		traffics = []
				#amount of rrhs
		rrhs_amount = 100
		#list of rrhs of the network
		rrhs = []



'''
#starts simulation
#simulation environment
env = simpy.Environment()
#create the graph
gp = g.createGraph()
#create the control plane
cp = Control_Plane(env, "Graph", gp, "EA", "fog_first")
#traffic generator
tg = Traffic_Generator(env,distribution, None, cp)
#create the rrhs
cp.createRRHs(g.rrhs_amount,env)
np.shuffle(g.rrhs)
#create fog nodes
g.addFogNodes(gp, g.fogs)
#add RRHs to the graph
g.addRRHs(gp, 0, 32, "0")
g.addRRHs(gp, 32, 64, "1")
g.addRRHs(gp, 64, 96, "2")
g.addRRHs(gp, 96, 128, "3")
g.addRRHs(gp, 128, 160, "4")
#g.addRRHs(gp, 0, 5, "0")
#g.addRRHs(gp, 5, 10, "1")
#g.addRRHs(gp, 10, 15, "2")
#g.addRRHs(gp, 15, 20, "3")
#g.addRRHs(gp, 20, 25, "4")
print(g.rrhs_fog)
#starts the simulation
#print(power_consumption)
#print(average_power_consumption)
env.run(until = 86401)

#for i in range(len(g.actives_rrhs)):
#	print(gp["s"]["RRH{}".format(i)]["capacity"])
#	print(nx.edges(gp, "RRH{}".format(i)))
#print(nx.edges(gp))
#print(gp["fog0"]["d"]["capacity"])
#neighbors = g.nx.all_neighbors(gp, "s")
#for i in neighbors:
#	print(i)
#print("Cost is {}".format(g.assignVPON(gp)))
#print(g.fog_rrhs)
'''
