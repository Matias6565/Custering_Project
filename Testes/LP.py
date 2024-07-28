#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import logging
#import cplex
import sys
import time
import random as rand
from docplex.mp.model import Model
import docplex.mp
from docplex.cp.model import *
import simpy
import functools
import random as np
import importlib
import csv
import numpy
import pandas as pd
import cplex

# Bloqueio para o restante sobrante. 1 req = 614.4
#total traffic resto da divisão por req = req bloq
# ==============================
__author__ = 'Matias Romário'
__email__ = "matiasrps@ufba.br/matiasromario@ieee.org"
__version__ = '3.0'
# ==============================

act_cloud = 0
act_fog = 0
act_lambda = 0

fog_delay = 0.0000980654 # 10km
#cloud_delay = 0.0001961308 # 20 km
#fog_delay = 0.000024516 # 5 km
cloud_delay = 0.000049033 # 10km 0,000033356
ONU_delay = 0.0000016 # 1,6 µs
LC_delay = 0.0000015 # 1,5 µs 0.000069283

# Taxa CPRI 
Band = 1000000 #Very big Num
#cpri = [614.4, 122.88, 460.8, 552.96, 0] # Valores do tráfego por split

#cpri = [1000, 166, 211, 240]#split = 0->8, 1->2, 2->6, 3->7
cpri = [1966, 74, 119, 674.4] 


#cpri = [1000, 1000]#C-RAN - Só nuvem
#cpri = [1000, 0]#Cf-RAN - Tudo nuvem ou Tudo Fog

#Use case: One-way latency DL bandwidth UL bandwidth.Source: 159_functional_splits_and_use_cases_for_sc_virtualization.pdf
#Latency_Req = [0.000050967, 0.000450967, 0.000450967, 0.001450967] # 1 ->C-RAN; 2 -> PHY; 3-> Split MAC ; 4-> PDCP-RLC. 0.00025 - cl_delay
Latency_Req = [0.0001, 0.0025, 0.0005, 0.00045] # 1 ->C-RAN; 2 -> PHY; 3-> Split MAC ; 4-> PDCP-RLC. 0.00025 - cl_delay
#Delay = [0.000049033, 0.000024516,0.000024516, 0.00098065, 0.00098065, 0.000049033, 0.000049033, 0.000049033]#Atraso de nó # Real
Delay = [0.000067033, 0.000024516,0.000024516, 0.000024516, 0.000024516]#Atraso de nó

cpri_rate = 614.4 # Taxa CPRI
wavelength_capacity = [10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0,10000.0, 10000.0, 10000.0, 10000.0, 10000.0]
lambda_state = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
switchBandwidth = [10000.0,10000.0,10000.0,10000.0,10000.0,10000.0,10000.0,10000.0,10000.0,10000.0]

switch_state = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
rrhs_on_nodes = [0,0,0,0,0,0,0]
#Amount_onus = []
split_state = [0,0,0,0,0,0,0,0]


# rrhs
rrhs = range(0,1)
# total de nós #TODO resolver conflito ao aumentar os nós
nodes = range(0, 3)#4
#Total de Split TODO: precisa de ajuste
Split= range(0, 4) # São 4
# total de lambdas
#lambdas = range(0, 8)
lambdas = range(0, 9)
node_capacity = [40000, 20000, 20000, 10000, 9830.4] # Para o CPRI
#node_capacity = [78640, 29490, 29490, 29490, 9830.4] # Novo - 40 rrhs no CPRI 0
#node_capacity = [60000, 10000, 10000, 10000, 10000]#cf-ran and split
#node_capacity = [60000, 0]
proc_delay = [0.00001, 0.00002, 0.00002, 0.00002, 0.00002]


# Custo dos Nós
nodeCost = [600.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0]
# Custo do line card
lc_cost = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
# Custo split
s_cost = [0.0, 20.0, 15.0, 10.0, 5.0]
#Custo por Onu ativada
Onu_cost = 7.7
RRH_cost = 20
nodeState = [0,0,0,0,0,0,0]

lambda_node = [
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
[1,1,1,1,1,1],
]


#Classe ILP
class LP(object):
	# Função para construir e resolver o modelo relaxado
	def buildModel_relaxed():
	    mdl = Model()
	    
	    x = mdl.binary_var_dict([(i, j, w, s) for i in rrhs for j in nodes for w in lambdas for s in Split], name='Rrh/Node/Lambda/Split')
	    k = mdl.binary_var_dict([(w, j, s) for w in lambdas for j in nodes for s in Split], name='Lambda/Nonde/Split')
	    y = mdl.continuous_var_dict([(i, j) for i in rrhs for j in nodes], name='RRH/Node')
	    xn = mdl.binary_var_dict([(j) for j in nodes], name='Nodes activated')
	    z = mdl.binary_var_dict([(w, j) for w in lambdas for j in nodes], name='Lambda/Node')
	    t = mdl.continuous_var_dict([(i, j, s) for i in rrhs for j in nodes for s in Split], name='RRH/Node/Split')
	    s = mdl.continuous_var_dict([(i, s) for i in rrhs for s in Split], name='Rhh/Split')
	    g = mdl.continuous_var_dict([(i, j, w, s) for i in rrhs for j in nodes for w in lambdas for s in Split], name='Redirections')
	    e = mdl.continuous_var_dict([(j) for j in nodes], name="Switch/Node")
	    
	    # Definindo as restrições
	    mdl.add_constraints(mdl.sum(x[i, j, w, s] for j in nodes for w in lambdas for s in Split) == 2 for i in rrhs)
	    mdl.add_constraints(mdl.sum(s[i, s] for s in Split) == 1 for i in rrhs)
	    mdl.add_constraints(mdl.sum(s[i, s]) <= 1.0 for s in Split for i in rrhs)
	    mdl.add_constraints(mdl.sum(x[i, j, w, s] * cpri[s] for s in Split for i in rrhs for j in nodes[0:1]) <= wavelength_capacity[w] for w in lambdas)
	    mdl.add_constraints(mdl.sum(x[i, j, w, s] * (cpri[0] - cpri[s]) for s in Split for i in rrhs for j in nodes[1:]) <= wavelength_capacity[w] for w in lambdas)
	    mdl.add_constraints(mdl.sum(x[i, j, w, s] * cpri[s] for s in Split for i in rrhs for w in lambdas) <= node_capacity[j] for j in nodes[0:1])
	    mdl.add_constraints(mdl.sum(x[i, j, w, s] * (cpri[0] - cpri[s]) for s in Split for i in rrhs for w in lambdas) <= node_capacity[j] for j in nodes[1:])
	    mdl.add_constraints(mdl.sum(y[i, j] for j in nodes) == 2 for i in rrhs)
	    mdl.add_constraints(y[i, 0] == 1 for i in rrhs)
	    mdl.add_constraints(y[i, j] <= mdl.sum(x[i, j, w, s] for s in Split for w in lambdas) for i in rrhs for j in nodes)
	    mdl.add_constraints(Band * y[i, j] >= mdl.sum(x[i, j, w, s] for s in Split for w in lambdas) for i in rrhs for j in nodes)
	    mdl.add_constraints(y[i, j] <= mdl.sum(t[i, j, s] for s in Split) for i in rrhs for j in nodes)
	    mdl.add_constraints(mdl.sum(y[i, j]) <= 1.0 for j in nodes for i in rrhs)
	    mdl.add_constraints(t[i, j, s] <= mdl.sum(s[i, s]) for s in Split for j in nodes for i in rrhs)
	    mdl.add_constraints(t[i, j, s] <= mdl.sum(x[i, j, w, s] for w in lambdas) for w in lambdas for j in nodes for i in rrhs for s in Split)
	    mdl.add_constraints(mdl.sum(t[i, j, s]) <= 1.0 for j in nodes for i in rrhs for s in Split)
	    mdl.add_constraints(Band * xn[j] >= mdl.sum(x[i, j, w, s] for i in rrhs for w in lambdas for s in Split) for j in nodes)
	    mdl.add_constraints(xn[j] <= mdl.sum(x[i, j, w, s] for i in rrhs for w in lambdas for s in Split) for j in nodes)
	    mdl.add_constraints(mdl.sum(z[w, j] for j in nodes) <= 1 for w in lambdas)
	    mdl.add_constraints(Band * z[w, j] >= mdl.sum(x[i, j, w, s] for i in rrhs for s in Split) for w in lambdas for j in nodes)
	    mdl.add_constraints(z[w, j] <= mdl.sum(x[i, j, w, s] for i in rrhs for s in Split) for w in lambdas for j in nodes)
	    mdl.add_constraints(mdl.sum(e[j]) <= 1.0 for j in nodes)
	    mdl.add_constraints(e[j] <= mdl.sum(y[i, j] for i in rrhs) for j in nodes)
	    
	    # Objetivo (exemplo, ajuste conforme necessário)
	    mdl.minimize(mdl.sum(y[i, j] for i in rrhs for j in nodes))
	    
	    return mdl


	#================ Solver ================
	def solveILP(self):
		self.mdl.minimize(self.mdl.sum(self.xn[j] * nodeCost[j] for j in self.nodes) + self.mdl.sum(self.z[w,j] * lc_cost[w] for w in self.lambdas for j in self.nodes) + self.mdl.sum(self.s[i,s] * s_cost[s] for s in self.Split for i in self.rrhs))

		self.mdl.parameters.lpmethod = 6
		self.mdl.parameters.timelimit = 500
		self.sol = self.mdl.solve()
		return self.sol



	#print
	def print_var_values_relaxed(self):
		for i in self.x:
			if self.x[i].solution_value >0:
				#self.x[i].solution_value == 1.0
				print("{} is {}".format(self.x[i], self.x[i].solution_value))

		for i in self.s:
			if self.s[i].solution_value >0:
				print("{} is {}".format(self.s[i], self.s[i].solution_value))
				#self.s[i].solution_value == round(self.s[i].solution_value,2)
				#print("{} is {}".format(self.s[i], round(self.s[i].solution_value,2)))

		for i in self.k:
			if self.k[i].solution_value >0:
				print("{} is {}".format(self.k[i], self.k[i].solution_value))

		for i in self.t:
			if self.t[i].solution_value >0:
				#self.t[i].update(1.0)
				print("{} is {}".format(self.t[i], self.t[i].solution_value))
				#print("{} is {}".format(self.t[i], round(self.t[i].solution_value,2)))

		for i in self.g:
			if self.g[i].solution_value >0:
				print("{} is {}".format(self.g[i], self.g[i].solution_value))

		for i in self.y:
			if self.y[i].solution_value >0:
				print("{} is {}".format(self.y[i], self.y[i].solution_value))
				#print("{} is {}".format(self.y[i], round(self.y[i].solution_value,2)))

		for i in self.e:
			if self.e[i].solution_value >0:
				print("{} is {}".format(self.e[i], self.e[i].solution_value))

		for i in self.xn:
			if self.xn[i].solution_value >0:
				print("{} is {}".format(self.xn[i], self.xn[i].solution_value))

		for i in self.z:
			if self.z[i].solution_value >0:
				print("{} is {}".format(self.z[i], self.z[i].solution_value))


	# Retornar relaxações1
	def return_solution_values_relaxed(self):
		self.var_x = []
		self.var_u = []
		self.var_y = []
		self.var_xn = []
		self.var_g = []
		self.var_k = []
		self.var_s = []
		self.var_t = []
		self.var_z = []
		#self.var_u = []
		self.var_g = []
		self.var_e = []


		for i in self.x:
			if self.x[i].solution_value > 0:
				self.var_x.append(i)

		for i in self.k:
			if self.k[i].solution_value > 0:
				self.var_k.append(i)

		for i in self.t:
			if self.t[i].solution_value > 0:
				self.var_t.append(i)

		for i in self.y:
			if self.y[i].solution_value > 0:
				self.var_y.append(i)

		for i in self.xn:
			if self.xn[i].solution_value > 0:
				self.var_xn.append(i)

		for i in self.s:
			if self.s[i].solution_value > 0:
				self.var_s.append(i)

		for i in self.e:
			if self.e[i].solution_value > 0:
				self.var_e.append(i)

		for i in self.g:
			if self.g[i].solution_value > 0:
				self.var_g.append(i)

		for i in self.t:
			if self.t[i].solution_value > 0:
				self.var_t.append(i)

		for i in self.z:
			if self.z[i].solution_value > 0:
				self.var_z.append(i)

		solution = Solution(self.var_x, self.var_z, self.var_k, self.var_s, self.var_y, self.var_xn, self.var_t, self.var_g, self.var_e)
		return solution



	# Função para verificar os resultados gerados em cada interação 0.0, 5.0, 4.0, 3.0, 2.0, 1.0
	def updateValues(self, solution):
		self.updateRRH(solution)
		for key in solution.var_x:
			node_id = key[1]
			lambda_id = key[2]
			split_id = key[3]
			rrhs_on_nodes[node_id] += 1
			if nodeState[node_id] == 0:
				nodeState[node_id] = 1
			if lambda_state[lambda_id] == 0:
				#delay_tot += LC_Delay
				lambda_state[lambda_id] = 1
			if split_state[split_id] == 0:
				split_state[split_id] = 1

	
	def update_splits(self, solution):
		self.updateRRH(solution)
		for i in solution.var_x:
			node_id = i[1]
			lambda_id = i[2]
			split_id = i[3]
			if split_id == 0 and node_id >=1:
				rrhs_on_nodes[node_id] -= 1
				if len(range(rrhs_on_nodes[node_id])) <= 0:
					nodeState[node_id] = 0
				if len(range(rrhs_on_nodes[node_id])) <= 0 and lambda_state[lambda_id] == 1 and wavelength_capacity[lambda_id] == 10000.0:
					split_state[split_id] = 0
					lambda_state[lambda_id] = 0
			if split_id == 7 and node_id ==0:
				rrhs_on_nodes[node_id] -= 1
				if len(range(rrhs_on_nodes[node_id])) == 0:
					nodeState[node_id] = 0
				if lambda_state[lambda_id] == 1 and wavelength_capacity[lambda_id] == 10000.0:
					split_state[split_id] = 0
					lambda_state[lambda_id] = 0
			else:
				pass



	# Banda direcionada para a Nuvem
	def Cloud_Band(self,solution):
		self.updateRRH(solution)
		total_traffic = 0.0
		for key in solution.var_x:
			node_id = key[1]
			split_id = key[3]
			if nodeState[node_id] ==1:
				if node_id == 0:
					total_traffic += cpri[split_id]
		return total_traffic

	def Fog_Band(self,solution):
		total_traffic = 0.0
		self.updateRRH(solution)
		for key in solution.var_x:
			node_id = key[1]
			split_id = key[3]
			if nodeState[node_id] ==1:
				if node_id >= 1:
					total_traffic += (cpri[0] - cpri[split_id])
		return total_traffic

	#put the solution values into the RRH
	def updateRRH(self,solution):
			for i in range(len(self.rrh)):
				self.rrh[i].var_x = solution.var_x[i]
				#self.rrh[i].var_t = solution.var_t[i]



# Função para arredondar a solução
def round_solution(mdl, solution):
    rounded_solution = {}
    for var in mdl.iter_variables():
        value = solution.get_value(var)
        if var.vartype in {'B', 'I'}:
            rounded_solution[var] = round(value)
        else:
            rounded_solution[var] = value
    return rounded_solution

# Função para verificar a validade da solução arredondada
def validate_solution(mdl, rounded_solution):
    for c in mdl.iter_constraints():
        if not c.is_satisfied(rounded_solution):
            return False
    return True

# Construir e resolver o modelo relaxado
mdl = buildModel_relaxed()
solution = mdl.solve()

# Arredondar a solução relaxada
rounded_solution = round_solution(mdl, solution)

# Verificar a validade da solução arredondada
is_valid = validate_solution(mdl, rounded_solution)

if is_valid:
    print("A solução arredondada é válida.")
else:
    print("A solução arredondada não é válida.")

