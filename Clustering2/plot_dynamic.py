# Plota
from scipy.ndimage.filters import gaussian_filter1d
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from scipy.stats import zscore
import numpy as np
from scipy.interpolate import interp1d
from sklearn import preprocessing


#df = pd.read_csv('ML.csv', sep=';', quoting=csv.QUOTE_NONE, encoding='utf-8')
df1 = pd.read_csv('ML170.csv', sep=';', quoting=csv.QUOTE_NONE, encoding='utf-8')
df2 = pd.read_csv('ML200.csv', sep=';', quoting=csv.QUOTE_NONE, encoding='utf-8')
df3 = pd.read_csv('ML300.csv', sep=';', quoting=csv.QUOTE_NONE, encoding='utf-8')

time = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

#y_time = gaussian_filter1d(df['CF-RAN_bloqued'], sigma=1)
#y2_time = gaussian_filter1d(df['C-RAN_bloqued'], sigma=1)
fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(df1['Latencia']*10000000, marker='.', color='black', linewidth=1, label='150 RRHs')  # Plot some data on the axes.
ax.plot(df2['Latencia']*10000000, marker='x', color='red', linewidth=1, label='200 RRHs')  # Plot some data on the axes.
ax.plot(df3['Latencia']*10000000, marker='8', color='blue', linewidth=1, label='300 CF-RAN')  # Plot some data on the axes.
ax.set_xlabel('Spent Hours')  # Add an x-label to the axes.
ax.set_ylabel('Latencia (μs)')  # Add a y-label to the axes.
#ax.set_title("Avg. Power Consumption")  # Add a title to the axes.
ax.legend()  # Add a legend.
#plt.grid()
plt.savefig('Latencia.png')
plt.show()


#y_time = gaussian_filter1d(df['CF-RAN_bloqued'], sigma=1)
#y2_time = gaussian_filter1d(df['C-RAN_bloqued'], sigma=1)
fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(df1['Energia'], marker='.', color='black', linewidth=1, label='150 RRHs')  # Plot some data on the axes.
ax.plot(df2['Energia'], marker='x', color='red', linewidth=1, label='200 RRHs')  # Plot some data on the axes.
ax.plot(df3['Energia'], marker='8', color='blue', linewidth=1, label='300 CF-RAN')  # Plot some data on the axes.
ax.set_xlabel('Spent Hours')  # Add an x-label to the axes.
ax.set_ylabel('Energia (Watts)')  # Add a y-label to the axes.
#ax.set_title("Avg. Power Consumption")  # Add a title to the axes.
ax.legend()  # Add a legend.
#plt.grid()
plt.savefig('Energia_cran.png')
plt.show()

#y_time = gaussian_filter1d(df['CF-RAN_bloqued'], sigma=1)
#y2_time = gaussian_filter1d(df['C-RAN_bloqued'], sigma=1)
fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(df1['Cobertura']*100, marker='.', color='black', linewidth=1, label='150 RRHs')  # Plot some data on the axes.
ax.plot(df2['Cobertura']*100, marker='x', color='red', linewidth=1, label='200 RRHs')  # Plot some data on the axes.
ax.plot(df3['Cobertura']*100, marker='8', color='blue', linewidth=1, label='300 CF-RAN')  # Plot some data on the axes.
ax.set_xlabel('Spent Hours')  # Add an x-label to the axes.
ax.set_ylabel('Cobertura (%)')  # Add a y-label to the axes.
#ax.set_title("Avg. Power Consumption")  # Add a title to the axes.
ax.legend()  # Add a legend.
#plt.grid()
plt.savefig('Energia_cran.png')
plt.show()


'''
df2 = df[df >0]
print(df['Latencia'].mean()*10000000)
print(df['Latencia'].max()*10000000)
print(df2['Latencia'].min()*10000000)
print(df['Latencia'].std()*10000000)

#y_time = gaussian_filter1d(df['CF-RAN_bloqued'], sigma=1)
#y2_time = gaussian_filter1d(df['C-RAN_bloqued'], sigma=1)
fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(df['Cobertura'], df['Latencia']*10000000, marker='.', color='black', linewidth=1, label='FS CF-RAN')  # Plot some data on the axes.
ax.set_xlabel('Spent Hours')  # Add an x-label to the axes.
ax.set_ylabel('Latencia (μs)')  # Add a y-label to the axes.
#ax.set_title("Avg. Power Consumption")  # Add a title to the axes.
ax.legend()  # Add a legend.
#plt.grid()
plt.savefig('Cobertura.png')
plt.show()


#y_time = gaussian_filter1d(df['CF-RAN_bloqued'], sigma=1)
#y2_time = gaussian_filter1d(df['C-RAN_bloqued'], sigma=1)
fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot( df['Energia'], df['Latencia']*10000000,marker='.', color='black', linewidth=1, label='FS CF-RAN')  # Plot some data on the axes.
ax.set_xlabel('Spent Hours')  # Add an x-label to the axes.
ax.set_ylabel('Latencia (μs)')  # Add a y-label to the axes.
#ax.set_title("Avg. Power Consumption")  # Add a title to the axes.
ax.legend()  # Add a legend.
#plt.grid()
plt.savefig('Eerngia.png')
plt.show()
'''
