import networkx as nx
import matplotlib.pyplot as plt
import math
import time
import random
import RRH
import graph_simulator as sim
import Util as u
import numpy as np
import pandas as pd
G = nx.DiGraph()
import re
import pickle
import Util as util

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from sklearn import cluster
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score

import os
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib






#=== Imput Parameters
#Não funciona com Listas - Adapta o split na execução...
ecpri_split = [1966, 74, 119, 675]
#ecpri_split = [614.4, 122.88, 460.8, 552.96]
Split_E = 0
Split_I = 0
Split_D = 0
Split_A = 0
Split_Tot = 0
ecpri = 1966
rrhs_amount = 80
lambda_capacity = 5 * ecpri
#fog capacity
fog_capacity = 10 * ecpri
#fog_capacity = 5 * ecpri # cenário do PLI
#fog_capacity = 20 * ecpri
#cloud capacity
cloud_capacity = 100 *ecpri
#cloud_capacity = 60 *ecpri #Cenário do PLI
#node power costs
fog_cost = 300
cloud_cost = 1
#number of fogs
#fogs = 8 #diamico
fogs = 5 #statico
#nodes costs
line_card_consumption = 20
costs = {}
for i in range(fogs):
	costs["fog{}".format(i)] = 300
costs["cloud"] = 600

#nodes capacities
capacities = {}
for i in range(fogs):
	capacities["fog{}".format(i)] = fog_capacity
capacities["cloud"] = cloud_capacity

limit = 0.7
#list of all rrhs
rrhs = []
#list of actives rrhs on the graph
actives_rrhs =[]
#list of available VPONs
available_vpons = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
cloud_wavelength = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
fog_wavelength = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]



#available_vpons = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
#cloud_wavelength = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#fog_wavelength = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]





vpons_capacity = {}

load_node = {}
load_node["cloud"] = 0.0
for i in range(fogs):
  load_node["fog_bridge{}".format(i)] = 0.0
#keeps the cost of the nodes

for i in range(20):
	vpons_capacity[i] = 10000

fog_rrhs = {}
for i in range(fogs):
	fog_rrhs["fog_bridge{}".format(i)] = []

rrhs_fog = {}
fogs_vpons = {}
for i in range(fogs):
	fogs_vpons["fog{}".format(i)] = []
cloud_vpons = []

fog_activated_rrhs= {}
for i in range(fogs):
  fog_activated_rrhs["fog_bridge{}".format(i)] = 0

rrhs_proc_node = {}
for i in range(rrhs_amount):
  rrhs_proc_node["RRH{}".format(i)] = None


#Randmizando a distância das antenas
#antenas2 = []
#ID = []
#inicio = 150
#fim = 220
#for t in range(200):
#  antenas2.append(random.randint(inicio, fim))
#  ID.append(t)
#print(antenas2)
#np.random.seed(1)
#dados = pd.DataFrame(data={"ID":ID, "RRHs": antenas2})
#dados.to_csv("Rural.csv", sep=';',index=False)

pred = []
#Split_E = []

dataset = []
ID = []
split_label = []
inicio = 0
fim = 450

#Delay_Split_E = 100
#Delay_Split_I = 250
#Delay_Split_D = 450
#Delay_Split_B = 500
'''
for t in range(2000):
  dados = random.randint(inicio, fim)
  dataset.append(dados)
  ID.append(t)
  if dados <=120:
    split_label.append('SpĺitE')
  if dados >120 and dados <=250:
    split_label.append('SpĺitI')
  if dados >250 and dados <=450:
    split_label.append('SpĺitD')
  if dados >450:
    split_label.append('SpĺitB')

#print(antenas2)
#np.random.seed(1)
data = pd.DataFrame(data={"ID":ID, "Latencia": dataset, "label":split_label})
data.to_csv("dataset2.csv", sep=';',index=False)
'''


#Delay parameters
fog_delay = 0.0000980654 #atraso de transmissão -- 5 km
cloud_delay = 0.0001961308 #atraso de transmissão -- 10km
delay_costs = {}
delay_costs["cloud"] = cloud_delay
for i in range(fogs):
	delay_costs["fog{}".format(i)] = fog_delay
Latency_Req = [0.0001, 0.0025, 0.0005, 0.0003] # 1 ->C-RAN; 2 -> PHY; 3-> Split MAC ; 4-> PDCP-RLC. 0.00025 - cl_delay



#--------------------------------------------------------------
#
#			Construção das Heurísticas
#
#--------------------------------------------------------------


#=============================================================================================================================
#------------------------------------------------- Clustering -


def Kmeans(t):
  #kmeans = KMeans(init="random", n_clusters=t, n_init=10, max_iter=300, random_state=42)
  k_clusters = t
  results = []
  print(n_clusters_)



def Clustering_Split_Latency_aware(graph):
  #Delay_Split_E = 100
  #Delay_Split_I = 250
  #Delay_Split_D = 450
  #Delay_Split_B = 500
  global Split_E , Split_I, Split_D, Split_B, Split_A, Split_Tot
  Fog_Proc_Delay = 2
  traffic_Cloud = 0 # pega a antena que vai transmitir o split e
  fog_traffic = 0 # pega a antena que vai transmitir o split I
  for i in range(len(actives_rrhs)):
    #print("Distacia da antena {} é igual a {}".format(i,u.distancia(i)))
    transmission_delay = 0.000005 * u.distancia(i)
    max_traffic = ecpri_split[0]*(i+1)
    atraso_total_nuvem = (u.atraso_fila() + transmission_delay)*1000000
    atraso_total_nuvem2 = (u.total_D(i,graph) + transmission_delay)*1000000
    #print("RRHs {} gerando {} com atraso de {}".format(i+1,(i+1)*1966, atraso_total_nuvem2))
    x = np.array(atraso_total_nuvem2)
    x2 = np.array(i)
    #print(x)
    #print(x2)
    x = np.vstack((x2, x)).T
    #print(x2)
    #loaded_model = pickle.load(open('kmean_clustering.pkl','rb'))
    loaded_model = joblib.load('kmeans.joblib')
    result2 = loaded_model.predict(x)
    #result2 = result2.replace('[','').replace(']','')
    #print(result2)
    #print(result2)
    if result2.item() == 'SpĺitE':
      Split_E+=1
      Split_I+=0
      Split_D+=0
      Split_A+=0
      Split_Tot+=1
      if cloud_capacity > traffic_Cloud: 
        traffic_Cloud += ecpri_split[0]
        if traffic_Cloud <= graph["cloud"]["d"]["capacity"]:
          if graph["bridge"]["cloud"]["capacity"] == 0 or traffic_Cloud > graph["bridge"]["cloud"]["capacity"]:
            while graph["bridge"]["cloud"]["capacity"] < traffic_Cloud:
              if cloud_wavelength:
                graph["bridge"]["cloud"]["capacity"] += 9824
                cloud_vpons.append(cloud_wavelength.pop())
              else:
                print("No VPON available!")
          else:
            pass#print("OKKKK")
        elif traffic_Cloud > graph["cloud"]["d"]["capacity"]:
          if cloud_wavelength:
            num_vpons = 0
            num_vpons = math.ceil(traffic_Cloud/lambda_capacity)
            while graph["bridge"]["cloud"]["capacity"] < graph["cloud"]["d"]["capacity"]:
              if cloud_wavelength:
                graph["bridge"]["cloud"]["capacity"] += 9824
                cloud_vpons.append(cloud_wavelength.pop())
                num_vpons -= 1
              else:
                print("No VPON available!")
                pass
      else:
        pass 

    #Split I
    elif result2.item() == 'SpĺitI':
      Split_I+=1
      Split_E+=0
      Split_D+=0
      Split_A+=0
      Split_Tot+=1
      #print("Passou")
      traffic_Cloud += ecpri_split[1]
      fog_traffic += ecpri_split[0] - ecpri_split[1]
      #print("fog traffic {}".format(fog_traffic))
      #print("Tráfego da nuvem {} e para a fog {}".format(traffic_Cloud, fog_traffic))
      if cloud_capacity > traffic_Cloud: 
        traffic_Cloud += ecpri_split[0]
        if traffic_Cloud <= graph["cloud"]["d"]["capacity"]:
          if graph["bridge"]["cloud"]["capacity"] == 0 or traffic_Cloud > graph["bridge"]["cloud"]["capacity"]:
            while graph["bridge"]["cloud"]["capacity"] < traffic_Cloud:
              if cloud_wavelength:
                graph["bridge"]["cloud"]["capacity"] += 9824
                cloud_vpons.append(cloud_wavelength.pop())
              else:
                print("No VPON available!")
          else:
            pass#print("OKKKK")
        elif traffic_Cloud > graph["cloud"]["d"]["capacity"]:
          if cloud_wavelength:
            num_vpons = 0
            num_vpons = math.ceil(traffic_Cloud/lambda_capacity)
            while graph["bridge"]["cloud"]["capacity"] < graph["cloud"]["d"]["capacity"]:
              if cloud_wavelength:
                graph["bridge"]["cloud"]["capacity"] += 9824
                cloud_vpons.append(cloud_wavelength.pop())
                num_vpons -= 1
              else:
                print("No VPON available!")
                pass
      else:
        pass 
      #residual = fog_traffic
      kk = math.ceil(fog_traffic/(fog_capacity))
      #print("Tráfego total {}, tráfego total na nuvem {} tráfego total na fog {} e total de kks {}".format(max_traffic, traffic_Cloud, fog_traffic, kk))
      #print("Total de fogs ativadas necessárias: {} Para {} tráfego em fog".format(kk,fog_traffic))
      while kk >0:
        if fog_traffic <= fog_capacity:
          k = 0
          if graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 0:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(fog_traffic/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 9824 and fog_traffic > 9824:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(fog_traffic/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 19648 and fog_traffic > 19648:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(fog_traffic/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 29472 and fog_traffic > 29472:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(fog_traffic/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
            kk-=1
        if fog_traffic > fog_capacity and fog_traffic <= 2*fog_capacity:
          residual = fog_traffic - fog_capacity
          k = 1
          if graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 0:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 9824 and residual > 9824:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 19648 and residual > 19648:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 29472 and residual > 29472:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
            kk-=1
        if fog_traffic > 2*fog_capacity and fog_traffic <= 3*fog_capacity:
          residual = fog_traffic - (2*fog_capacity)
          k = 2
          if graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 0:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 9824 and residual > 9824:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 19648 and residual > 19648:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 29472 and residual > 29472:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
            kk-=1
        if fog_traffic > 3*fog_capacity and fog_traffic <= 4*fog_capacity:
          residual = fog_traffic - (3*fog_capacity)
          k = 3
          if graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 0:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 9824 and residual > 9824:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 19648 and residual > 19648:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 29472 and residual > 29472:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
            kk-=1
        if fog_traffic > 4*fog_capacity and fog_traffic <= 5*fog_capacity:
          residual = fog_traffic - (4*fog_capacity)
          k = 4
          if graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 0:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 9824 and residual > 9824:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 19648 and residual > 19648:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 29472 and residual > 29472:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
            kk-=1
        if fog_traffic > 5*fog_capacity:
          print("Sem recursos")
          kk-=1
          break
        else:
          break


    #Split D
    elif result2.item() == 'SpĺitD':
      Split_D+=1
      Split_Tot+=1
      Split_E+=0
      Split_I+=0
      Split_A+=0
      traffic_Cloud += ecpri_split[2]
      fog_traffic += ecpri_split[0] - ecpri_split[2]
      #print("fog traffic {}".format(fog_traffic))
      #print("Tráfego da nuvem {} e para a fog {}".format(traffic_Cloud, fog_traffic))
      if cloud_capacity > traffic_Cloud: 
        traffic_Cloud += ecpri_split[0]
        if traffic_Cloud <= graph["cloud"]["d"]["capacity"]:
          if graph["bridge"]["cloud"]["capacity"] == 0 or traffic_Cloud > graph["bridge"]["cloud"]["capacity"]:
            while graph["bridge"]["cloud"]["capacity"] < traffic_Cloud:
              if cloud_wavelength:
                graph["bridge"]["cloud"]["capacity"] += 9824
                cloud_vpons.append(cloud_wavelength.pop())
              else:
                print("No VPON available!")
          else:
            pass#print("OKKKK")
        elif traffic_Cloud > graph["cloud"]["d"]["capacity"]:
          if cloud_wavelength:
            num_vpons = 0
            num_vpons = math.ceil(traffic_Cloud/lambda_capacity)
            while graph["bridge"]["cloud"]["capacity"] < graph["cloud"]["d"]["capacity"]:
              if cloud_wavelength:
                graph["bridge"]["cloud"]["capacity"] += 9824
                cloud_vpons.append(cloud_wavelength.pop())
                num_vpons -= 1
              else:
                print("No VPON available!")
                pass
      else:
        pass 
      #residual = fog_traffic
      kk = math.ceil(fog_traffic/(fog_capacity))
      #print("Tráfego total {}, tráfego total na nuvem {} tráfego total na fog {} e total de kks {}".format(max_traffic, traffic_Cloud, fog_traffic, kk))
      #print("Total de fogs ativadas necessárias: {} Para {} tráfego em fog".format(kk,fog_traffic))
      while kk >0:
        if fog_traffic <= fog_capacity:
          k = 0
          if graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 0:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(fog_traffic/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 9824 and fog_traffic > 9824:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(fog_traffic/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 19648 and fog_traffic > 19648:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(fog_traffic/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 29472 and fog_traffic > 29472:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(fog_traffic/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
            kk-=1
        if fog_traffic > fog_capacity and fog_traffic <= 2*fog_capacity:
          residual = fog_traffic - fog_capacity
          k = 1
          if graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 0:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 9824 and residual > 9824:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 19648 and residual > 19648:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 29472 and residual > 29472:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
            kk-=1
        if fog_traffic > 2*fog_capacity and fog_traffic <= 3*fog_capacity:
          residual = fog_traffic - (2*fog_capacity)
          k = 2
          if graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 0:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 9824 and residual > 9824:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 19648 and residual > 19648:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 29472 and residual > 29472:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
            kk-=1
        if fog_traffic > 3*fog_capacity and fog_traffic <= 4*fog_capacity:
          residual = fog_traffic - (3*fog_capacity)
          k = 3
          if graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 0:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 9824 and residual > 9824:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 19648 and residual > 19648:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 29472 and residual > 29472:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
            kk-=1
        if fog_traffic > 4*fog_capacity and fog_traffic <= 5*fog_capacity:
          residual = fog_traffic - (4*fog_capacity)
          k = 4
          if graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 0:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 9824 and residual > 9824:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 19648 and residual > 19648:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 29472 and residual > 29472:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
            kk-=1
        if fog_traffic > 5*fog_capacity:
          print("Sem recursos")
          kk-=1
          break
        else:
          break


    #Split A
    elif result2.item() == 'SplitA':
      Split_A+=1
      Split_Tot+=1
      Split_E+=0
      Split_I+=0
      Split_D+=0
      traffic_Cloud += 0
      fog_traffic += ecpri_split[0]
      #print("fog traffic {}".format(fog_traffic))
      #print("Tráfego da nuvem {} e para a fog {}".format(traffic_Cloud, fog_traffic))
      #residual = fog_traffic
      kk = math.ceil(fog_traffic/(fog_capacity))
      #print("Tráfego total {}, tráfego total na nuvem {} tráfego total na fog {} e total de kks {}".format(max_traffic, traffic_Cloud, fog_traffic, kk))
      #print("Total de fogs ativadas necessárias: {} Para {} tráfego em fog".format(kk,fog_traffic))
      while kk >0:
        if fog_traffic <= fog_capacity:
          k = 0
          if graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 0:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(fog_traffic/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 9824 and fog_traffic > 9824:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(fog_traffic/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 19648 and fog_traffic > 19648:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(fog_traffic/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 29472 and fog_traffic > 29472:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(fog_traffic/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
            kk-=1
        if fog_traffic > fog_capacity and fog_traffic <= 2*fog_capacity:
          residual = fog_traffic - fog_capacity
          k = 1
          if graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 0:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 9824 and residual > 9824:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 19648 and residual > 19648:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 29472 and residual > 29472:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
            kk-=1
        if fog_traffic > 2*fog_capacity and fog_traffic <= 3*fog_capacity:
          residual = fog_traffic - (2*fog_capacity)
          k = 2
          if graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 0:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 9824 and residual > 9824:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 19648 and residual > 19648:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 29472 and residual > 29472:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
            kk-=1
        if fog_traffic > 3*fog_capacity and fog_traffic <= 4*fog_capacity:
          residual = fog_traffic - (3*fog_capacity)
          k = 3
          if graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 0:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 9824 and residual > 9824:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 19648 and residual > 19648:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 29472 and residual > 29472:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
            kk-=1
        if fog_traffic > 4*fog_capacity and fog_traffic <= 5*fog_capacity:
          residual = fog_traffic - (4*fog_capacity)
          k = 4
          if graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 0:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 9824 and residual > 9824:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 19648 and residual > 19648:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
          elif graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"]== 29472 and residual > 29472:
            graph["fog_bridge{}".format(k)]["fog{}".format(k)]["capacity"] += 9824
            if fog_wavelength:
              f = math.ceil(residual/(lambda_capacity)) 
              fogs_vpons["fog{}".format(f)].append(fog_wavelength.pop())
              f -=1
            kk-=1
        if fog_traffic > 5*fog_capacity:
          print("Sem recursos")
          kk-=1
          break
        else:
          break
    else:
      print("Aqui Sem VPON - e sem split")
      break



















def Clustering_Split_Latency_aware2(graph):
  #Delay_Split_E = 100
  #Delay_Split_I = 250
  #Delay_Split_D = 450
  #Delay_Split_B = 500
  global Split_E , Split_I, Split_D, Split_B, Split_A, Split_Tot
  Fog_Proc_Delay = 2
  traffic_Cloud = 0 # pega a antena que vai transmitir o split e
  fog_traffic = 0 # pega a antena que vai transmitir o split I
  for i in range(len(actives_rrhs)):
    #print("Distacia da antena {} é igual a {}".format(i,u.distancia(i)))
    transmission_delay = 0.000005 * u.distancia(i)
    max_traffic = ecpri_split[0]*(i+1)
    atraso_total_nuvem = (u.atraso_fila() + transmission_delay)*1000000
    atraso_total_nuvem2 = (u.total_D(i,graph) + transmission_delay)*1000000
    #print("RRHs {} gerando {} com atraso de {}".format(i+1,(i+1)*1966, atraso_total_nuvem2))
    x = np.array(atraso_total_nuvem2)
    x2 = np.array(i)
    #print(x)
    #print(x2)
    x = np.vstack((x2, x)).T
    #print(x2)
    #loaded_model = pickle.load(open('kmean_clustering.pkl','rb'))
    loaded_model = joblib.load('kmeans.joblib')
    result2 = loaded_model.predict(x)
    #result2 = result2.replace('[','').replace(']','')
    #print(result2)
    #print(result2)
    if result2.item() == 'SplitE':
        Split_E += 1
        Split_I += 0
        Split_D += 0
        Split_A += 0
        Split_Tot += 1

        # Lógica para desalocar VPONs não utilizadas
        #excess_vpons = graph["bridge"]["cloud"]["capacity"] - traffic_Cloud
        if excess_vpons > 0:
            for _ in range(excess_vpons):
                if cloud_vpons:
                    graph["bridge"]["cloud"]["capacity"] -= 9824
                    cloud_wavelength.append(cloud_vpons.pop())
                else:
                    print("No VPON available!")

        # Lógica para alocar VPONs para a nuvem, se necessário
        if cloud_capacity > traffic_Cloud:
            traffic_Cloud += ecpri_split[0]
            if traffic_Cloud <= graph["cloud"]["d"]["capacity"]:
                if graph["bridge"]["cloud"]["capacity"] == 0 or traffic_Cloud > graph["bridge"]["cloud"]["capacity"]:
                    while graph["bridge"]["cloud"]["capacity"] < traffic_Cloud:
                        if cloud_wavelength:
                            graph["bridge"]["cloud"]["capacity"] += 9824
                            cloud_vpons.append(cloud_wavelength.pop())
                        else:
                            print("No VPON available!")
                else:
                    pass
            elif traffic_Cloud > graph["cloud"]["d"]["capacity"]:
                if cloud_wavelength:
                    num_vpons = math.ceil(traffic_Cloud / lambda_capacity)
                    while graph["bridge"]["cloud"]["capacity"] < graph["cloud"]["d"]["capacity"]:
                        if cloud_wavelength:
                            graph["bridge"]["cloud"]["capacity"] += 9824
                            cloud_vpons.append(cloud_wavelength.pop())
                            num_vpons -= 1
                        else:
                            print("No VPON available!")
                            pass
        else:
            pass

    elif result2.item() == 'SplitI':
        Split_I += 1
        Split_E += 0
        Split_D += 0
        Split_A += 0
        Split_Tot += 1

        # Lógica para desalocar VPONs não utilizadas
        excess_vpons = graph["bridge"]["cloud"]["capacity"] - traffic_Cloud
        if excess_vpons > 0:
            for _ in range(excess_vpons):
                if cloud_vpons:
                    graph["bridge"]["cloud"]["capacity"] -= 9824
                    cloud_wavelength.append(cloud_vpons.pop())
                else:
                    print("No VPON available!")

        # Lógica para alocar VPONs para a nuvem, se necessário
        if cloud_capacity > traffic_Cloud:
            traffic_Cloud += ecpri_split[0]
            if traffic_Cloud <= graph["cloud"]["d"]["capacity"]:
                if graph["bridge"]["cloud"]["capacity"] == 0 or traffic_Cloud > graph["bridge"]["cloud"]["capacity"]:
                    while graph["bridge"]["cloud"]["capacity"] < traffic_Cloud:
                        if cloud_wavelength:
                            graph["bridge"]["cloud"]["capacity"] += 9824
                            cloud_vpons.append(cloud_wavelength.pop())
                        else:
                            print("No VPON available!")
                else:
                    pass
            elif traffic_Cloud > graph["cloud"]["d"]["capacity"]:
                if cloud_wavelength:
                    num_vpons = math.ceil(traffic_Cloud / lambda_capacity)
                    while graph["bridge"]["cloud"]["capacity"] < graph["cloud"]["d"]["capacity"]:
                        if cloud_wavelength:
                            graph["bridge"]["cloud"]["capacity"] += 9824
                            cloud_vpons.append(cloud_wavelength.pop())
                            num_vpons -= 1
                        else:
                            print("No VPON available!")
                            pass
        else:
            pass

        # Lógica para alocar VPONs para os nós de fog, se necessário
        residual = fog_traffic
        for k in range(len(fogs)):
            if residual <= 0:
                break
            elif residual <= fog_capacity:
                if graph[f"fog_bridge{k}"][f"fog{k}"]["capacity"] == 0:
                    graph[f"fog_bridge{k}"][f"fog{k}"]["capacity"] += 9824
                    if fog_wavelength:
                        fogs_vpons[f"fog{k}"].append(fog_wavelength.pop())
                        residual -= lambda_capacity
                elif graph[f"fog_bridge{k}"][f"fog{k}"]["capacity"] == 9824 and residual > 0:
                    graph[f"fog_bridge{k}"][f"fog{k}"]["capacity"] += 9824
                    if fog_wavelength:
                        fogs_vpons[f"fog{k}"].append(fog_wavelength.pop())
                        residual -= lambda_capacity
            else:
                print("No resources for Fog {}".format(k))
                break

    elif result2.item() == 'SplitD':
        Split_D += 1
        Split_Tot += 1
        Split_E += 0
        Split_I += 0
        Split_A += 0
        traffic_Cloud += ecpri_split[2]
        fog_traffic += ecpri_split[0] - ecpri_split[2]

        # Lógica para desalocar VPONs não utilizadas
        excess_vpons = graph["bridge"]["cloud"]["capacity"] - traffic_Cloud
        if excess_vpons > 0:
            for _ in range(excess_vpons):
                if cloud_vpons:
                    graph["bridge"]["cloud"]["capacity"] -= 9824
                    cloud_wavelength.append(cloud_vpons.pop())
                else:
                    print("No VPON available!")

        # Lógica para alocar VPONs para a nuvem, se necessário
        if cloud_capacity > traffic_Cloud:
            traffic_Cloud += ecpri_split[0]
            if traffic_Cloud <= graph["cloud"]["d"]["capacity"]:
                if graph["bridge"]["cloud"]["capacity"] == 0 or traffic_Cloud > graph["bridge"]["cloud"]["capacity"]:
                    while graph["bridge"]["cloud"]["capacity"] < traffic_Cloud:
                        if cloud_wavelength:
                            graph["bridge"]["cloud"]["capacity"] += 9824
                            cloud_vpons.append(cloud_wavelength.pop())
                        else:
                            print("No VPON available!")
                else:
                    pass
            elif traffic_Cloud > graph["cloud"]["d"]["capacity"]:
                if cloud_wavelength:
                    num_vpons = math.ceil(traffic_Cloud / lambda_capacity)
                    while graph["bridge"]["cloud"]["capacity"] < graph["cloud"]["d"]["capacity"]:
                        if cloud_wavelength:
                            graph["bridge"]["cloud"]["capacity"] += 9824
                            cloud_vpons.append(cloud_wavelength.pop())
                            num_vpons -= 1
                        else:
                            print("No VPON available!")
                            pass
        else:
            pass

        # Lógica para alocar VPONs para os nós de fog, se necessário
        kk = math.ceil(fog_traffic / fog_capacity)
        while kk > 0:
            for k in range(len(fogs)):
                if fog_traffic <= fog_capacity:
                    if graph[f"fog_bridge{k}"][f"fog{k}"]["capacity"] == 0:
                        graph[f"fog_bridge{k}"][f"fog{k}"]["capacity"] += 9824
                        if fog_wavelength:
                            f = math.ceil(fog_traffic / lambda_capacity)
                            fogs_vpons[f"fog{k}"].append(fog_wavelength.pop())
                            f -= 1
                    elif graph[f"fog_bridge{k}"][f"fog{k}"]["capacity"] == 9824 and fog_traffic > 9824:
                        graph[f"fog_bridge{k}"][f"fog{k}"]["capacity"] += 9824
                        if fog_wavelength:
                            f = math.ceil(fog_traffic / lambda_capacity)
                            fogs_vpons[f"fog{k}"].append(fog_wavelength.pop())
                            f -= 1
                    kk -= 1
                if fog_traffic > fog_capacity * (k + 1):
                    residual = fog_traffic - fog_capacity * (k + 1)
                    if graph[f"fog_bridge{k}"][f"fog{k}"]["capacity"] == 0:
                        graph[f"fog_bridge{k}"][f"fog{k}"]["capacity"] += 9824
                        if fog_wavelength:
                            f = math.ceil(residual / lambda_capacity)
                            fogs_vpons[f"fog{k}"].append(fog_wavelength.pop())
                            f -= 1
                    elif graph[f"fog_bridge{k}"][f"fog{k}"]["capacity"] == 9824 and residual > 9824:
                        graph[f"fog_bridge{k}"][f"fog{k}"]["capacity"] += 9824
                        if fog_wavelength:
                            f = math.ceil(residual / lambda_capacity)
                            fogs_vpons[f"fog{k}"].append(fog_wavelength.pop())
                            f -= 1
                    kk -= 1
                if fog_traffic > fog_capacity * (len(fogs)):
                    print("Sem recursos")
                    kk -= 1
                    break
                else:
                    break
































#----------------------------------------------------
#remove unnecessary bandwidth(VPONs) from the links (fronthaul and fog nodes)
def removeVPON(graph):
  traffic = len(actives_rrhs) * ecpri
  if traffic <= (limit * graph["cloud"]["d"]["capacity"]) and traffic <= graph["bridge"]["cloud"]["capacity"]:
    #cloud can handle the traffic, if there are VPONs on fogs, release them
    for i in range(fogs):
      if graph["fog_bridge{}".format(i)]["fog{}".format(i)]["capacity"] > 0:
        graph["fog_bridge{}".format(i)]["fog{}".format(i)]["capacity"] = 0
        while fogs_vpons["fog{}".format(i)]:
          #print("Poping")
          fog_wavelength.append(fogs_vpons["fog{}".format(i)].pop())
    #release the VPONs of the fronthaul until only the necessary to support current traffic is reached
    num_vpons = 0
    num_vpons = (math.ceil(traffic/lambda_capacity))*9824
    while graph["bridge"]["cloud"]["capacity"] > num_vpons:
      #print("Releasing the cloud")
      graph["bridge"]["cloud"]["capacity"] -= 9824
      cloud_wavelength.append(cloud_vpons.pop())


#remove unnecessary bandwidth(VPONs) from the links (fronthaul and fog nodes)
def removeSplitVPON(graph):
  traffic = len(actives_rrhs) * ecpri
  if traffic <= graph["cloud"]["d"]["capacity"] and traffic <= graph["bridge"]["cloud"]["capacity"]:
    for i in range(fogs):
      if graph["fog_bridge{}".format(i)]["fog{}".format(i)]["capacity"] > 0:
        graph["fog_bridge{}".format(i)]["fog{}".format(i)]["capacity"] = 0
        while fogs_vpons["fog{}".format(i)]:
          fog_wavelength.append(fogs_vpons["fog{}".format(i)].pop())
    #release the VPONs of the fronthaul until only the necessary to support current traffic is reached
    num_vpons = 0
    num_vpons = (math.ceil(traffic/lambda_capacity))*9824
    while graph["bridge"]["cloud"]["capacity"] > num_vpons:
      graph["bridge"]["cloud"]["capacity"] -= 9824
      cloud_wavelength.append(cloud_vpons.pop())


#Totally random remove VPONs
def randomRemoveVPONs(graph):
  #gets the traffic
  traffic = getIncomingTraffic()
  total_bd = getTotalBandwidth(graph)
  need_vpons = math.ceil(traffic/lambda_capacity)
  current_vpons = round(total_bd/lambda_capacity)
  #now, removes the VPONs until the available bandwidth is equal to
  while current_vpons > need_vpons:
    #print("Try to remove")
    node = getRandomNode()
    if node == "cloud":
      if graph["bridge"]["cloud"]["capacity"] > 0:
        graph["bridge"]["cloud"]["capacity"] -= 9824
        available_vpons.append(cloud_vpons.pop())
        total_bd = getTotalBandwidth(graph)
        current_vpons = round(total_bd/lambda_capacity)
        #print("Removing {}".format(node))
    else:
      bridge = getFogBridge(graph, node)
      if graph[bridge][node]["capacity"] > 0:
        graph[bridge][node]["capacity"] -= 9824
        available_vpons.append(fogs_vpons[node].pop())
        total_bd = getTotalBandwidth(graph)
        current_vpons = round(total_bd/lambda_capacity)
        #print("Removing {}".format(node))


# Remove VPONs desnecessários dos links (fronthaul e nós de fog)
def removeSplitVPON2(graph):
    traffic = len(actives_rrhs) * ecpri

    # Verifica se o tráfego é menor ou igual à capacidade da nuvem e do link bridge->cloud
    if traffic <= graph["cloud"]["d"]["capacity"] and traffic <= graph["bridge"]["cloud"]["capacity"]:
        # Reseta a capacidade dos nós de fog para 0
        for i in range(len(fogs)):
            if graph[f"fog_bridge{i}"][f"fog{i}"]["capacity"] > 0:
                graph[f"fog_bridge{i}"][f"fog{i}"]["capacity"] = 0
                while fogs_vpons[f"fog{i}"]:
                    fog_wavelength.append(fogs_vpons[f"fog{i}"].pop())

        # Calcula o número necessário de VPONs para suportar o tráfego atual
        num_vpons = (math.ceil(traffic / lambda_capacity)) * 9824

        # Libera VPONs do fronthaul até alcançar somente o necessário para suportar o tráfego atual
        while graph["bridge"]["cloud"]["capacity"] > num_vpons:
            graph["bridge"]["cloud"]["capacity"] -= 9824
            cloud_wavelength.append(cloud_vpons.pop())


def assignVPON(graph):
  traffic = 0
  #calculate the total incoming traffic
  traffic = len(actives_rrhs) * ecpri
  #verify if the cloud alone can support this traffic
  if traffic <= graph["cloud"]["d"]["capacity"]:
    #print("Cloud Can Handle It!")
    #verify if the fronthaul has lambda. If so, does nothing, otherwise, put the necessary vpons
    if graph["bridge"]["cloud"]["capacity"] == 0 or traffic > graph["bridge"]["cloud"]["capacity"]:
      #print("Putting VPONs on Fronthaul")
      #calculate the VPONs necessaries and put them on the fronthaul
      #num_vpons = 0
      #num_vpons = math.ceil(traffic/lambda_capacity)
      #for i in range(num_vpons):
      #  graph["bridge"]["cloud"]["capacity"] += 9824
      #  allocated_vpons.append(available_vpons.pop())
      #this ways seems better than the above method
      while graph["bridge"]["cloud"]["capacity"] < traffic:
        if available_vpons:
          graph["bridge"]["cloud"]["capacity"] += 9824
          cloud_vpons.append(available_vpons.pop())
        else:
          print("No VPON available!")
    else:
      pass#print("OKKKK")
  elif traffic > graph["cloud"]["d"]["capacity"]:
    if available_vpons:
      #calculate the amount necessary on VPONs and put the maximum on the cloud and the rest on the fogs
      num_vpons = 0
      num_vpons = math.ceil(traffic/lambda_capacity)
      while graph["bridge"]["cloud"]["capacity"] < graph["cloud"]["d"]["capacity"] :
        if available_vpons:
          graph["bridge"]["cloud"]["capacity"] += 9824
          cloud_vpons.append(available_vpons.pop())
          num_vpons -= 1
        else:
            print("No VPON available!")
      #First-Fit Fog VPON Allocation - When there is VPON and traffic is greater than the total available bandwidth, put it on the next Fog Node
      while available_vpons:
        for i in range(fogs):#this is the First-Fit Fog VPON Allocation
          if available_vpons:
            graph["fog_bridge{}".format(i)]["fog{}".format(i)]["capacity"] += 9824 
            fogs_vpons["fog{}".format(i)].append(available_vpons.pop())##available_vpons
            num_vpons -= 1
          else:
              print("No VPON available!")
    #else: print('No available VPONs')
  #return traffic
  #print("Cloud VPONs: {}".format(cloud_vpons))
  #print("Fogs VPONs: {}".format(fogs_vpons))

def removeFogFirstVPON(graph):
  #get the bandwidth available on midhaul
  traffic = getIncomingTraffic()
  midhaul_bd = 0.0
  md = getMidhaulBandiwdth(graph)
  for i in range(fogs):
    midhaul_bd += md["fog{}".format(i)]
  #check if the midhaul can handle the traffic
  if traffic <= midhaul_bd:
    #turn the VPONs of cloud off
    while graph["bridge"]["cloud"]["capacity"] > 0:
      graph["bridge"]["cloud"]["capacity"] -= 9824
      available_vpons.append(cloud_vpons.pop())
      #print("CLEANING CLOUD")
  #now, check if some fog node must be turned off
  for i in range(fogs):
    fog_traffic = getRRHsFogLoad(graph, "fog{}".format(i))
    if fog_traffic == 0 and graph[getFogBridge(graph, "fog{}".format(i))]["fog{}".format(i)]["capacity"] > 0:
      #print("CLEANING FOG{}".format(i))
      graph[getFogBridge(graph, "fog{}".format(i))]["fog{}".format(i)]["capacity"] = 0
      available_vpons.append(fogs_vpons["fog{}".format(i)].pop())


#--------------------------------------------------------------
#
#			Criando grafos
#
#--------------------------------------------------------------


'''
As seguintes funções estão relacionadas à criação dos grafos:
RRHs, Nós de processamento, Ligações e custos dos arcos 

'''

#========================
#create rrhs

def createGraph():
    G = nx.DiGraph()
    G.add_edges_from([("bridge", "cloud", {'capacity': 0, 'weight': cloud_cost}),
                    ("cloud", "d", {'capacity': cloud_capacity, 'weight': 0}),
                   ])
    return G
def createRRHs():
  for i in range(rrhs_amount):
    rrhs.append(RRH.RRH(ecpri, i))
    
#create graph
G.add_edges_from([("bridge", "cloud", {'capacity': 0, 'weight': cloud_cost}),
                    ("cloud", "d", {'capacity': cloud_capacity, 'weight': 0}),
                   ])    
    


def addRRHs(graph, bottom, rrhs, fog):
    for i in range(bottom,rrhs):
      addNodesEdgesSet(graph, "RRH{}".format(i), "{}".format(fog))
      rrhs_fog["RRH{}".format(i)] = "fog_bridge{}".format(fog)

def addFogNodes(graph, fogs):
    for i in range(fogs):
      graph.add_edges_from([("fog_bridge{}".format(i), "fog{}".format(i), {'capacity': 0, 'weight':fog_cost}),
                          ("fog{}".format(i), "d", {'capacity': fog_capacity, 'weight':0}),
                          ])


'''
def addNodesEdgesSet(graph, node, fog_bridge):
    graph.add_edges_from([("s", node, {'capacity': 0, 'weight': 0}),
      (node, "fog_bridge{}".format(fog_bridge), { 'weight': 0}),
      (node, "bridge", { 'weight': 0}),
      ])
    indexRRH(node, fog_bridge)
'''
'''
def indexRRH(node, fog_bridge):
  fog_rrhs["fog_bridge{}".format(fog_bridge)].append(node)    
'''


def addNodesEdgesSet(graph, node, fog_bridge):
    # Adicione 'node' aos nós do gráfico
    graph.add_node(node)

    # Verifique se a chave 'fog_bridgeX' existe no dicionário 'fog_rrhs' e crie-a se não existir
    if 'fog_bridge{}'.format(fog_bridge) not in fog_rrhs:
        fog_rrhs['fog_bridge{}'.format(fog_bridge)] = []

    # Adicione as arestas ao gráfico
    graph.add_edges_from([("s", node, {'capacity': 0, 'weight': 0}),
      (node, "fog_bridge{}".format(fog_bridge), { 'weight': 0}),
      (node, "bridge", { 'weight': 0}),
      ])
    
    # Indexe 'node' no dicionário 'fog_rrhs'
    indexRRH(node, fog_bridge)

def indexRRH(node, fog_bridge):
    # Verifique se a chave 'fog_bridgeX' existe no dicionário 'fog_rrhs' e crie-a se não existir
    if 'fog_bridge{}'.format(fog_bridge) not in fog_rrhs:
        fog_rrhs['fog_bridge{}'.format(fog_bridge)] = []

    # Adicione 'node' à lista correspondente no dicionário
    fog_rrhs["fog_bridge{}".format(fog_bridge)].append(node)
    
def startNode(graph, node):
  graph["s"][node]["capacity"] = ecpri

def endNode(graph, node):
  graph["s"][node]["capacity"] = 0.0


#update the amount of activated RRHs attached to a fog node
def addActivated(rrh):
  fog_activated_rrhs[rrhs_fog[rrh]] += 1


#remove traffic from RRH from its processing node
def removeRRHNode(rrh):
  global load_node, rrhs_proc_node
  load_node[rrhs_proc_node[rrh]] -= ecpri
  rrhs_proc_node[rrh] = None

#update the amount of activated RRHs attached to a fog node
def minusActivated(rrh):
  if fog_activated_rrhs[rrhs_fog[rrh]] > 0:
    fog_activated_rrhs[rrhs_fog[rrh]] -= 1

#update load on processing node
def update_node_load(node, load):
  load_node[node] += load
  load_node[node] = round(load_node[node],1)


#get all transmitted traffic from source to destination
def getTransmittedTraffic(mincostFlow):
  transmitted = 0.0
  transmitted += mincostFlow["cloud"]["d"]
  for i in range(fogs):
    transmitted += mincostFlow["fog{}".format(i)]["d"]
  return transmitted

#get the amount of activated RRHs on each fog node
def activatedFogRRHs():
  return fog_activated_rrhs

#get all incoming traffic
def getIncomingTraffic():
  return ecpri * (len(actives_rrhs))


#--------------------------------------------------------------
#
#			Funções Util
#
#--------------------------------------------------------------

#return the total available bandwidth on all network links
def getTotalBandwidth(graph):
  total = 0.0
  total += graph["bridge"]["cloud"]["capacity"]
  for i in range(fogs):
    total += graph["fog_bridge{}".format(i)]["fog{}".format(i)]["capacity"]
  return total

#get the blocking probability of each executed max flow min cost
def getBlockingProbability(mincostFlow):
  lost_traffic = (len(actives_rrhs)*ecpri) - getTransmittedTraffic(mincostFlow)
  blocking_probability = lost_traffic/(len(actives_rrhs)*ecpri)
  return blocking_probability

#check in which node the rrhs is being processed
def getProcessingNodes(graph, mincostFlow, rrh):
  #now, iterate over the flow of each neighbor of the RRH
  if mincostFlow[rrh][rrhs_fog[rrh]] != 0:
    update_node_load(rrhs_fog[rrh], ecpri)
    rrhs_proc_node[rrh] = rrhs_fog[rrh]
    #print("Inserted on fog")
    return True
  elif mincostFlow[rrh]["bridge"] != 0:
    update_node_load("cloud", ecpri)
    rrhs_proc_node[rrh] = "cloud"
    #print("Inserted on cloud")
    return True
  else:
    return False
  #print(mincostFlow[actives_rrhs[i]]["bridge"])


def Energy(graph):
  power_cost = 0.0
  if graph["bridge"]["cloud"]["capacity"] > 0:
    power_cost += 600
  else:
    power_cost +=300  

  return power_cost
  


#calculate the power consumption considering as active every node that has a VPON assigned, regardless if it is transmitting traffic or not
def Energy2(graph):
  power_cost = 0.0
  #if graph.get("fog_bridge{}".format(i), {}).get("fog{}".format(i), {}).get("capacity") > 0:
  if graph["bridge"]["cloud"]["capacity"] > 0:
    power_cost += costs["cloud"]
  for i in range(fogs):
    if graph["fog_bridge{}".format(i)]["fog{}".format(i)]["capacity"] > 0:
      power_cost += costs["fog{}".format(i)]
    power_cost += getBandwidthPower(graph)
  return power_cost
  
  
  
  
#calculate the power consumption considering as active every node that has a VPON assigned, regardless if it is transmitting traffic or not
def overallPowerConsumption2(graph):
  power_cost = 0.0
  if graph["bridge"]["cloud"]["capacity"] > 0:
    power_cost += costs["cloud"]
  for i in range(fogs):
    if graph["fog_bridge{}".format(i)]["fog{}".format(i)]["capacity"] > 0:
      power_cost += costs["fog{}".format(i)]
  power_cost += getBandwidthPower(graph)
  return power_cost
  


# Calcula o consumo de energia considerando como ativo todo nó que tem uma VPON atribuída e está transmitindo tráfego
def overallPowerConsumption(graph):
    power_cost = 0.0

    # Verifica se o link bridge->cloud tem capacidade > 0 (indicando VPON atribuída)
    if graph["bridge"]["cloud"]["capacity"] > 0:
        power_cost += costs["cloud"]

    # Itera sobre os nós de fog para verificar se têm capacidade > 0 (indicando VPON atribuída)
    for i in range(len(fogs)):
        if graph[f"fog_bridge{i}"][f"fog{i}"]["capacity"] > 0:
            power_cost += costs[f"fog{i}"]

    # Adiciona o custo de energia associado à largura de banda utilizada
    power_cost += getBandwidthPower(graph)

    return power_cost



#get the blocking probability of each executed max flow min cost
def getBlockingProbability(mincostFlow):
  lost_traffic = (len(actives_rrhs)*cpri_line) - getTransmittedTraffic(mincostFlow)
  blocking_probability = lost_traffic/(len(actives_rrhs)*cpri_line)
  return blocking_probability

#get the blocking probability of each executed max flow min cost
def get_TotLambdasNecessary(mincostFlow):
  tot_traffic = (len(actives_rrhs)*ecpri)
  lambdas_usage = tot_traffic/10000
  return lambdas_usage


#calculate the power consumed by active VPONs
def getBandwidthPower(graph):
  bandwidth_power = 0.0
  #first get the consumption of the fronthaul
  if graph["bridge"]["cloud"]["capacity"] > 0:
    bandwidth_power += (graph["bridge"]["cloud"]["capacity"] / 9824) * line_card_consumption
  #now, if there are VPONs on the fog nodes, calculates their power consumption
  for i in range(fogs):
    print("Total de {}".format(fogs))
    #if graph["fog_bridge{}".format(i)]["fog{}".format(i)]["capacity"] > 0:
    #  bandwidth_power += (graph["fog_bridge{}".format(i)]["fog{}".format(i)]["capacity"] / 9824) * line_card_consumption
  return bandwidth_power

