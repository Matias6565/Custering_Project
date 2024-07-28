import networkx as nx
#import matplotlib.pyplot as plt
import math
import time
import random
import Latency_Aware as g
import graph_simulator as sim
import csv
import pandas as pd
import numpy as np

G = nx.DiGraph()

#ecpri = [1966, 74, 119, 674.4] 
ecpri = 1966
rrhs_amount = 160
lambda_capacity = 16 * ecpri
#fog capacity
fog_capacity = 16 * ecpri
#cloud capacity
cloud_capacity = 80 *ecpri
line_card_consumption = 20
actives_rrhs =[]

costs = {}
fog_cost = 300
cloud_cost = 1
#number of fogs
fogs = 5

for i in range(fogs):
  costs["fog{}".format(i)] = 300
costs["cloud"] = 600
#nodes capacities
capacities = {}
for i in range(fogs):
  capacities["fog{}".format(i)] = fog_capacity
capacities["cloud"] = cloud_capacity

fog_delay = 0.0000980654
cloud_delay = 0.0001961308
delay_costs = {}
delay_costs["cloud"] = cloud_delay
for i in range(fogs):
  delay_costs["fog{}".format(i)] = fog_delay
#keeps the cloud and fog links to random VPON assignment
random_nodes = []
random_nodes.append("cloud")
for i in range(fogs):
  random_nodes.append("fog{}".format(i))


myfilepath = "Urban.csv"

## ======================================================= Results ======================================================= ##

#calculate the power consumption considering as active every node that has a VPON assigned, regardless if it is transmitting traffic or not
def overallPowerConsumption(graph):
  power_cost = 0.0
  if graph["bridge"]["cloud"]["capacity"] > 0:
    power_cost += costs["cloud"]
  for i in range(fogs):
    if graph["fog_bridge{}".format(i)]["fog{}".format(i)]["capacity"] > 0:
      power_cost += costs["fog{}".format(i)]
  power_cost += getBandwidthPower(graph)
  return power_cost

#calculate the power consumed by active VPONs
def getBandwidthPower(graph):
  bandwidth_power = 0.0
  #first get the consumption of the fronthaul
  if graph["bridge"]["cloud"]["capacity"] > 0:
    bandwidth_power += (graph["bridge"]["cloud"]["capacity"] / 9824) * line_card_consumption
  #now, if there are VPONs on the fog nodes, calculates their power consumption
  for i in range(fogs):
    if graph["fog_bridge{}".format(i)]["fog{}".format(i)]["capacity"] > 0:
      bandwidth_power += (graph["fog_bridge{}".format(i)]["fog{}".format(i)]["capacity"] / 9824) * line_card_consumption
  return bandwidth_power


#return the total available bandwidth on all network links
def getTotalBandwidth(graph):
  total = 0.0
  total += graph["bridge"]["cloud"]["capacity"]
  for i in range(fogs):
    total += graph["fog_bridge{}".format(i)]["fog{}".format(i)]["capacity"]
  return total


def getCloudBandwidth(graph):
  total = 0.0
  total += graph["bridge"]["cloud"]["capacity"]
  return total


def getFogBandwidth(graph):
  total = 0.0
  for i in range(fogs):
    total += graph["fog_bridge{}".format(i)]["fog{}".format(i)]["capacity"]
  return total


#calculates the average minimum delay on the network, taking in consider only the delay of the active nodes
def overallDelay(graph):
  total_delay = 0.0
  amount = 0
  if graph["bridge"]["cloud"]["capacity"] > 0:
    total_delay += cloud_delay
    amount += 1
  for i in range(fogs):
    if graph["fog_bridge{}".format(i)]["fog{}".format(i)]["capacity"] > 0:
      total_delay += fog_delay
      amount += 1
  total_delay = (total_delay/amount)
  return total_delay

#get the blocking probability of each executed max flow min cost
def getBlockingProbability(mincostFlow):
  if len(actives_rrhs)*ecpri <=0:
    blocking_probability = 0
  else:
    lost_traffic = (len(actives_rrhs)*ecpri) - g.getTransmittedTraffic(mincostFlow)
    blocking_probability = lost_traffic/(len(actives_rrhs)*ecpri)
  return blocking_probability


#get the blocking probability of each executed max flow min cost
def get_CloudLambdasNecessary(graph):
  total = 0.0
  total += graph["bridge"]["cloud"]["capacity"]
  total = math.ceil(total/10000)
  return total

def get_Fog_LambdasNecessary(graph,i):
  total = 0.0
  for i in range(fogs):
    total += graph["fog_bridge{}".format(i)]["fog{}".format(i)]["capacity"]
  total = math.ceil(total/10000)
  return total


#Pega a latência em todos os nós 
def atraso_fila():
  arrival_rate = sim.arrival_rate
  #print("avg. arrival_rate {}".format(sim.arrival_rate))
  cloud_service_rate = 0.00001
  fog_service_rate = 0.00002
  total_wait = 0
  queue_size = 0
  mu = 1.0/cloud_service_rate
  l = 1.0/arrival_rate
  rho = l/mu
  W = rho/mu/(1-rho)  # average weight in the queue
  T = 1/mu/(1-rho)    # average total system time.
  nq_bar = rho/(1.0 - rho) - rho # The average number waiting in the queue
  if W > 0:
    return W
  else:
    return 0

def distancia(value):
  df = pd.read_csv('Urban.csv', delimiter = ';')  
  val = df['RRHs'].values[value]
  return val

#Pega a latência em todos os nós 
def total_D(i,graph):
  arrival_rate = sim.arrival_rate
  cloud_service_rate = 0.00001
  if (graph["bridge"]["cloud"]["capacity"]) > 0:
    roh = (i*1966)/graph["bridge"]["cloud"]["capacity"]
  else:
    roh = 0
  total_wait = 0
  queue_size = 0
  mu = 1.0/cloud_service_rate
  l = 1.0/arrival_rate
  rho = l/mu
  #segundoW = 1/(cloud_service_rate - (1-roh))
  segundW= roh/mu/(1-roh)
  W = rho/mu/(1-rho)  # average weight in the queue
  T = 1/mu/(1-rho)    # average total system time.
  nq_bar = rho/(1.0 - rho) - rho # The average number waiting in the queue
  if segundW > 0:
    return segundW
  else:
    return 0


#Pega a latência em todos os nós 
def Total_Queue_Delay_dinamico(i,graph):
  i = i.replace("RRH", "");
  i = int(i)
  arrival_rate = sim.arrival_rate
  cloud_service_rate = 0.00001
  if graph["bridge"]["cloud"]["capacity"] > 0:
    roh = (i*1966)/graph["bridge"]["cloud"]["capacity"]
  else:
    roh = 0
  total_wait = 0
  queue_size = 0
  mu = 1.0/cloud_service_rate
  l = 1.0/arrival_rate
  rho = l/mu
  #segundoW = 1/(cloud_service_rate - (1-roh))
  segundW= roh/mu/(1-roh)
  W = rho/mu/(1-rho)  # average weight in the queue
  T = 1/mu/(1-rho)    # average total system time.
  nq_bar = rho/(1.0 - rho) - rho # The average number waiting in the queue
  if segundW > 0:
    return segundW
  else:
    return 0

#Pega a latência em todos os nós 
def Total_Delay(i,graph):
  i = i.replace("RRH", "");
  i = int(i)
  arrival_rate = sim.arrival_rate
  cloud_service_rate = 0.00001
  if graph["bridge"]["cloud"]["capacity"] > 0:
    roh = (i*1966)/graph["bridge"]["cloud"]["capacity"]
  else:
    roh = 0
  if graph["fog_bridge{}".format(1)]["fog{}".format(1)]["capacity"] > 0:
    roh2 = (i*1200)/graph["fog_bridge{}".format(1)]["fog{}".format(1)]["capacity"]
  else:
    roh2 = 0 
  total_wait = 0
  queue_size = 0
  mu = 1.0/cloud_service_rate
  l = 1.0/arrival_rate
  rho = l/mu
  #segundoW = 1/(cloud_service_rate - (1-roh))
  segundW= roh/mu/(1-roh)
  terceiroW = roh2/mu/(1-roh)
  W = rho/mu/(1-rho)  # average weight in the queue
  T = 1/mu/(1-rho)    # average total system time.
  nq_bar = rho/(1.0 - rho) - rho # The average number waiting in the queue
  tot = segundW + terceiroW + (distancia(i)*0.000005)
  if segundW > 0:
    return tot
  else:
    return distancia(i)*0.000005

