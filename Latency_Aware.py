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



#=== Imput Parameters
#Não funciona com Listas - Adapta o split na execução...
ecpri_split = [1966, 74, 119, 675]
ecpri = 1966
rrhs_amount = 80
lambda_capacity = 5 * ecpri
#fog capacity
#fog_capacity = 10 * ecpri
#fog_capacity = 5 * ecpri # cenário do PLI
fog_capacity = 10 * ecpri
#cloud capacity
cloud_capacity = 20 *ecpri
#cloud_capacity = 60 *ecpri #Cenário do PLI
#node power costs
fog_cost = 400
cloud_cost = 1
#number of fogs
#fogs = 8 #diamico
fogs = 5 #statico
#nodes costs
costs = {}
for i in range(fogs):
	costs["fog{}".format(i)] = 300
costs["cloud"] = 600

#nodes capacities
capacities = {}
for i in range(fogs):
	capacities["fog{}".format(i)] = fog_capacity
capacities["cloud"] = cloud_capacity

limit = 0.85
#list of all rrhs
rrhs = []
#list of actives rrhs on the graph
actives_rrhs =[]
#list of available VPONs
#available_vpons = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
available_vpons = [0,1,2,3,4,5,6,7,8,9,10]
#cloud_wavelength = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
#fog_wavelength = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]

cloud_wavelength = [0,1,2,3,4,5,6,7,8]
fog_wavelength = [0,1,2,3,4,5,6,7,8]

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
#inicio = 15
#fim = 30
#for t in range(200):
#  antenas2.append(random.randint(inicio, fim))
#  ID.append(t)
#print(antenas2)
#np.random.seed(1)
#dados = pd.DataFrame(data={"ID":ID, "RRHs": antenas2})
#dados.to_csv("antenas2.csv", sep=';',index=False)




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

#_______________________________________________________________________
import pandas as pd
import csv
import numpy as np

data_delay = []
data_maxTraffic = []
data_cloud_capacity = []
data_fog_capacity = []
data_cloud_vpons = []
data_fog_vpons = []
data_actives_rrhs = []


def Latency_aware(graph):
  Delay_Split_E = 100
  Delay_Split_I = 250
  Delay_Split_D = 450
  Delay_Split_B = 500
  Fog_Proc_Delay = 2
  traffic_Cloud = 0 # pega a antena que vai transmitir o split e
  fog_traffic = 0 # pega a antena que vai transmitir o split I
  Fog_Proc_Delay = 2
  traffic_Cloud = 0 # pega a antena que vai transmitir o split e
  fog_traffic = 0 # pega a antena que vai transmitir o split I
  for i in range(len(actives_rrhs)):
    data_actives_rrhs.append(actives_rrhs)
    transmission_delay = 0.000005 * u.distancia(i)
    #transmission_delay = 0.000005 * 10
    max_traffic = ecpri_split[0]*(i+1)
    data_maxTraffic.append(max_traffic)
    atraso_total_nuvem = (u.atraso_fila() + transmission_delay)*1000000
    data_delay.append(atraso_total_nuvem)

    #np.random.seed(1)
    #dados = pd.DataFrame(data={ "Atraso": data_delay, "Tráfego" : data_maxTraffic})
    #dados.to_csv("test.csv", sep=';',index=False,mode='a')

    #print("RRHs {} gerando {} com atraso de {}".format(i+1,(i+1)*1966, atraso_total_nuvem))
    #print("RRHs {} gerando {} com atraso de {}".format(i,i*1966, atraso_total_nuvem))
    if atraso_total_nuvem <= Delay_Split_E or i*1966 <cloud_capacity:
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
    elif (atraso_total_nuvem > Delay_Split_E and atraso_total_nuvem <= Delay_Split_I) or i*1966 <cloud_capacity:
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
    elif (atraso_total_nuvem > Delay_Split_I and atraso_total_nuvem <= Delay_Split_D) or i*1966 <cloud_capacity:
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
    elif (atraso_total_nuvem > Delay_Split_D or cloud_capacity < traffic_Cloud)  or cloud_capacity < traffic_Cloud:
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

def allocate_traffic_to_fog(fog_traffic, fog_capacity, lambda_capacity, fog_wavelength, fogs_vpons, graph, k):
    if k >= len(graph):
        return fog_traffic

    if fog_traffic <= 0:
        return fog_traffic

    capacity_key = f"fog_bridge{k}"
    fog_key = f"fog{k}"
    if fog_traffic <= fog_capacity:
        if graph[capacity_key][fog_key]["capacity"] == 0:
            graph[capacity_key][fog_key]["capacity"] += 9824
            if fog_wavelength:
                f = math.ceil(fog_traffic / lambda_capacity)
                fogs_vpons[fog_key].append(fog_wavelength.pop())
                print(f"Allocated VPON to {fog_key}, remaining fog_traffic: {fog_traffic}")
            else:
                print("No VPON available!")
        elif graph[capacity_key][fog_key]["capacity"] == 9824 and fog_traffic > 9824:
            graph[capacity_key][fog_key]["capacity"] += 9824
            if fog_wavelength:
                f = math.ceil(fog_traffic / lambda_capacity)
                fogs_vpons[fog_key].append(fog_wavelength.pop())
                print(f"Allocated VPON to {fog_key}, remaining fog_traffic: {fog_traffic}")
            else:
                print("No VPON available!")
        elif graph[capacity_key][fog_key]["capacity"] == 19648 and fog_traffic > 19648:
            graph[capacity_key][fog_key]["capacity"] += 9824
            if fog_wavelength:
                f = math.ceil(fog_traffic / lambda_capacity)
                fogs_vpons[fog_key].append(fog_wavelength.pop())
                print(f"Allocated VPON to {fog_key}, remaining fog_traffic: {fog_traffic}")
            else:
                print("No VPON available!")
        elif graph[capacity_key][fog_key]["capacity"] == 29472 and fog_traffic > 29472:
            graph[capacity_key][fog_key]["capacity"] += 9824
            if fog_wavelength:
                f = math.ceil(fog_traffic / lambda_capacity)
                fogs_vpons[fog_key].append(fog_wavelength.pop())
                print(f"Allocated VPON to {fog_key}, remaining fog_traffic: {fog_traffic}")
            else:
                print("No VPON available!")
        return fog_traffic

    residual = fog_traffic - fog_capacity
    graph[capacity_key][fog_key]["capacity"] += 9824
    if fog_wavelength:
        f = math.ceil(fog_capacity / lambda_capacity)
        fogs_vpons[fog_key].append(fog_wavelength.pop())
        print(f"Allocated VPON to {fog_key}, residual traffic: {residual}")
    else:
        print("No VPON available!")

    return allocate_traffic_to_fog(residual, fog_capacity, lambda_capacity, fog_wavelength, fogs_vpons, graph, k + 1)

def Latency_aware2(graph):
  Delay_Split_E = 100
  Delay_Split_I = 250
  Delay_Split_D = 450
  Delay_Split_B = 500
  Fog_Proc_Delay = 2
  traffic_Cloud = 0 # pega a antena que vai transmitir o split e
  fog_traffic = 0 # pega a antena que vai transmitir o split I
  Fog_Proc_Delay = 2
  traffic_cloud = 0 # pega a antena que vai transmitir o split e
  fog_traffic = 0 # pega a antena que vai transmitir o split I
  LAMBDA_CAPACITY = 9824

  for i in range(len(actives_rrhs)):
      data_actives_rrhs.append(actives_rrhs)
      transmission_delay = 0.000005 * u.distancia(i)
      max_traffic = ecpri_split[0] * (i + 1)
      data_maxTraffic.append(max_traffic)
      total_cloud_delay = (u.atraso_fila() + transmission_delay) * 1000000
      data_delay.append(total_cloud_delay)

      if total_cloud_delay <= Delay_Split_E or i * 1966 < cloud_capacity:
          traffic_cloud += allocate_traffic_to_cloud(traffic_cloud, ecpri_split[0], graph)
      elif Delay_Split_E < total_cloud_delay <= Delay_Split_I or i * 1966 < cloud_capacity:
          traffic_cloud += ecpri_split[1]
          fog_traffic += ecpri_split[0] - ecpri_split[1]
          allocate_traffic_to_fog(fog_traffic, fog_capacity, LAMBDA_CAPACITY, fog_wavelength, fogs_vpons, graph, 0)
      elif Delay_Split_I < total_cloud_delay <= Delay_Split_D or i * 1966 < cloud_capacity:
          traffic_cloud += ecpri_split[2]
          fog_traffic += ecpri_split[0] - ecpri_split[2]
          allocate_traffic_to_fog(fog_traffic, fog_capacity, LAMBDA_CAPACITY, fog_wavelength, fogs_vpons, graph, 0)
      else:
          fog_traffic = ecpri_split[0]
          allocate_traffic_to_fog(fog_traffic, fog_capacity, LAMBDA_CAPACITY, fog_wavelength, fogs_vpons, graph, 0)

#--------------------------------------------------------------
#
#			Heuristicas de Alocação
#
#--------------------------------------------------------------

def allocate_traffic_to_fog2(fog_traffic, fog_capacity, lambda_capacity, fog_wavelength, fogs_vpons, graph, k):
    if k >= len(graph):
        return fog_traffic

    if fog_traffic <= 0:
        return fog_traffic

    capacity_key = f"fog_bridge{k}"
    fog_key = f"fog{k}"
    if fog_traffic <= fog_capacity:
        if graph[capacity_key][fog_key]["capacity"] == 0:
            graph[capacity_key][fog_key]["capacity"] += 9824
            if fog_wavelength:
                f = math.ceil(fog_traffic / lambda_capacity)
                fogs_vpons[fog_key].append(fog_wavelength.pop())
        elif graph[capacity_key][fog_key]["capacity"] == 9824 and fog_traffic > 9824:
            graph[capacity_key][fog_key]["capacity"] += 9824
            if fog_wavelength:
                f = math.ceil(fog_traffic / lambda_capacity)
                fogs_vpons[fog_key].append(fog_wavelength.pop())
        elif graph[capacity_key][fog_key]["capacity"] == 19648 and fog_traffic > 19648:
            graph[capacity_key][fog_key]["capacity"] += 9824
            if fog_wavelength:
                f = math.ceil(fog_traffic / lambda_capacity)
                fogs_vpons[fog_key].append(fog_wavelength.pop())
        elif graph[capacity_key][fog_key]["capacity"] == 29472 and fog_traffic > 29472:
            graph[capacity_key][fog_key]["capacity"] += 9824
            if fog_wavelength:
                f = math.ceil(fog_traffic / lambda_capacity)
                fogs_vpons[fog_key].append(fog_wavelength.pop())
        return fog_traffic

    residual = fog_traffic - fog_capacity
    graph[capacity_key][fog_key]["capacity"] += 9824
    if fog_wavelength:
        f = math.ceil(fog_capacity / lambda_capacity)
        fogs_vpons[fog_key].append(fog_wavelength.pop())

    return allocate_traffic_to_fog(residual, fog_capacity, lambda_capacity, fog_wavelength, fogs_vpons, graph, k + 1)

def allocate_traffic_to_cloud(traffic_cloud, ecpri_split, graph):
    LAMBDA_CAPACITY = 9824
    cloud_capacity_key = "cloud"

    if cloud_capacity > traffic_cloud:
        traffic_cloud += ecpri_split
        if traffic_cloud <= graph[cloud_capacity_key]["d"]["capacity"]:
            while graph["bridge"]["cloud"]["capacity"] < traffic_cloud:
                if cloud_wavelength:
                    graph["bridge"]["cloud"]["capacity"] += LAMBDA_CAPACITY
                    cloud_vpons.append(cloud_wavelength.pop())
                else:
                    print("No VPON available!")
        else:
            num_vpons = math.ceil(traffic_cloud / LAMBDA_CAPACITY)
            while graph["bridge"]["cloud"]["capacity"] < graph[cloud_capacity_key]["d"]["capacity"]:
                if cloud_wavelength:
                    graph["bridge"]["cloud"]["capacity"] += LAMBDA_CAPACITY
                    cloud_vpons.append(cloud_wavelength.pop())
                    num_vpons -= 1
                else:
                    print("No VPON available!")
    return ecpri_split

def allocate_traffic_to_fog(fog_traffic, fog_capacity, lambda_capacity, fog_wavelength, fogs_vpons, graph, k):
    if k >= len(graph):
        return fog_traffic

    if fog_traffic <= 0:
        return fog_traffic

    capacity_key = f"fog_bridge{k}"
    fog_key = f"fog{k}"

    if fog_traffic <= fog_capacity:
        if graph[capacity_key][fog_key]["capacity"] == 0:
            graph[capacity_key][fog_key]["capacity"] += lambda_capacity
            if fog_wavelength:
                fogs_vpons[fog_key].append(fog_wavelength.pop())
        elif graph[capacity_key][fog_key]["capacity"] <= 3 * lambda_capacity and fog_traffic > graph[capacity_key][fog_key]["capacity"]:
            graph[capacity_key][fog_key]["capacity"] += lambda_capacity
            if fog_wavelength:
                fogs_vpons[fog_key].append(fog_wavelength.pop())
        return fog_traffic

    residual = fog_traffic - fog_capacity
    graph[capacity_key][fog_key]["capacity"] += lambda_capacity
    if fog_wavelength:
        fogs_vpons[fog_key].append(fog_wavelength.pop())

    return allocate_traffic_to_fog(residual, fog_capacity, lambda_capacity, fog_wavelength, fogs_vpons, graph, k + 1)





#--------------------------------------------------------------
#
#			Heuristicas de desalocação
#
#--------------------------------------------------------------


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
def createRRHs():
  for i in range(rrhs_amount):
    rrhs.append(RRH.RRH(ecpri, i))

def addRRHs(graph, bottom, rrhs, fog):
    for i in range(bottom,rrhs):
      addNodesEdgesSet(graph, "RRH{}".format(i), "{}".format(fog))
      rrhs_fog["RRH{}".format(i)] = "fog_bridge{}".format(fog)

def addFogNodes(graph, fogs):
    for i in range(fogs):
      graph.add_edges_from([("fog_bridge{}".format(i), "fog{}".format(i), {'capacity': 0, 'weight':fog_cost}),
                          ("fog{}".format(i), "d", {'capacity': fog_capacity, 'weight':0}),
                          ])



def addNodesEdgesSet(graph, node, fog_bridge):
    graph.add_edges_from([("s", node, {'capacity': 0, 'weight': 0}),
      (node, "fog_bridge{}".format(fog_bridge), { 'weight': 0}),
      (node, "bridge", { 'weight': 0}),
      ])
    indexRRH(node, fog_bridge)

def createGraph():
    G = nx.DiGraph()
    G.add_edges_from([("bridge", "cloud", {'capacity': 0, 'weight': cloud_cost}),
                    ("cloud", "d", {'capacity': cloud_capacity, 'weight': 0}),
                   ])
    return G

def indexRRH(node, fog_bridge):
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
    if graph["fog_bridge{}".format(i)]["fog{}".format(i)]["capacity"] > 0:
      bandwidth_power += (graph["fog_bridge{}".format(i)]["fog{}".format(i)]["capacity"] / 9824) * line_card_consumption
  return bandwidth_power

################################################################
'''import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
Loading the Data
data = pd.read_csv('Countryclusters.csv')
data
￼
Plotting the data
plt.scatter(data['Longitude'],data['Latitude'])
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()
￼
Selecting the feature
 x = data.iloc[:,1:3] # 1t for rows and second for columns
x
￼
Clustering
kmeans = KMeans(3)
means.fit(x)
Clustering Results
identified_clusters = kmeans.fit_predict(x)
identified_clusters
array([1, 1, 0, 0, 0, 2])
data_with_clusters = data.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(data_with_clusters['Longitude'],data_with_clusters['Latitude'],c=data_with_clusters['Clusters'],cmap='rainbow')
￼
Trying different method ( to find no .of clusters to be selected)
WCSS and Elbow Method
wcss=[]
for i in range(1,7):
kmeans = KMeans(i)
'''





















