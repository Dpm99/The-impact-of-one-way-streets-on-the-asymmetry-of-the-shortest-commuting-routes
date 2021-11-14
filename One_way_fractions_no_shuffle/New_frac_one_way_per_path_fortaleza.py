import osmnx as ox
import igraph as ig
import pandas as pd
import networkx as nx
import numpy as np
import multiprocessing as mp
import time
import random as rd
from itertools import product
import ast
from numpy import inf
import matplotlib.pyplot as plt
import collections

from IPython.display import clear_output

ox.config(log_console=True, use_cache=True, cache_folder = '/home/diogo_mota/Dropbox/Diogo_Rede_simetria/Beta_Experiment/Network_Data') ## mudar aqui o caminho

city_name = 'Fortaleza'

cities = ox.geocode_to_gdf(['Brasil, Fortaleza', 'Brasil, Caucaia', 'Brasil, Maracanaú', 'Brasil, Maracanaú',
                           'Brasil, Eusébio', 'Brasil, Itaitinga', 'Brasil Pacatuba'])

whole_polygon = cities.unary_union #unary union of both geometries
city_pol = cities['geometry'].iloc[0] #geometry of just the city

G_1 = ox.graph_from_polygon(city_pol, network_type='drive', simplify=True)
nodes_1, edges_1 = ox.graph_to_gdfs(G_1, nodes=True, edges=True)

G = ox.graph_from_polygon(whole_polygon, network_type='drive', simplify=True)
G_nx = nx.relabel.convert_node_labels_to_integers(G)
nodes, edges = ox.graph_to_gdfs(G_nx, nodes=True, edges=True)

weight = 'length'


edge_list = list(G_nx.edges())
length_list = list(nx.get_edge_attributes(G_nx, 'length').values())
key_list = list(edges['key'])

attr_list = [(edge_list[i], round(length_list[i],0), key_list[i]) for i in range(len(edge_list))]

bool_list = [] ## create the bools using my criteria
for item in attr_list:
    
    tup = item[0]
    length_tup = item[1]
    key_val = item[2]
    
    inv_tup = tup[::-1]
    value = (inv_tup, length_tup, key_val)
    
    if value in attr_list:
        bool_list.append(False) ## not one way
    else:
        bool_list.append(True) ## one way


round_length_list = [round(value,0) for value in length_list]
full_attr_df = pd.DataFrame(zip(edge_list, round_length_list, key_list, bool_list),
                           columns = ['Tuple', 'Length', 'K', 'Oneway'])
full_attr_df['highway'] = edges.highway

lanes_list = list(edges.lanes)
clean_lanes = []
for val in lanes_list:
    
    if type(val)!=list:
        if type(val)==float:
            clean_lanes.append(-1)
        else:
            clean_lanes.append(int(val))
    else:
        int_list = []
        for num in val:
            if type(num)==float:
                in_list.append(-1)
            else:
                int_list.append(int(num))
        clean_lanes.append(int_list)
        
full_attr_df['lanes'] = clean_lanes



new_oneway = []
for i in range(len(full_attr_df)):
    
    if type(full_attr_df['highway'].iloc[i])!=list and type(full_attr_df['lanes'].iloc[i])!=list:
        ## se quisermos as marginais como two way entao adicionar a esta condicao se for unclassified
        if full_attr_df['highway'].iloc[i] == 'unclassified' and full_attr_df['lanes'].iloc[i]>=3:
            new_oneway.append(False)

        elif full_attr_df['highway'].iloc[i] == 'trunk' or full_attr_df['highway'].iloc[i] == 'primary' or full_attr_df['highway'].iloc[i] == 'motorway':
            new_oneway.append(False)

        else:
            new_oneway.append(full_attr_df['Oneway'].iloc[i])
            
    else:
        #print(full_attr_df['highway'].iloc[i])
        if 'trunk' in full_attr_df['highway'].iloc[i]:
            new_oneway.append(False)
        elif 'primary' in full_attr_df['highway'].iloc[i]:
            new_oneway.append(False)
        elif 3 in full_attr_df['lanes'] or 4 in full_attr_df['lanes'] or 5 in full_attr_df['lanes']:
            new_oneway.append(False)
        else: new_oneway.append(True)
            
full_attr_df['New oneway'] = new_oneway


#


clean_full_attr_df = full_attr_df.drop(['Oneway', 'highway', 'lanes'], axis=1)




paths = pd.read_csv('/home/diogo_mota/Dropbox/Diogo_Rede_simetria/Beta_Experiment/'+city_name+'/'+city_name+'_100k_paths_no_shuffle.dat', index_col=0) ## mudar aqui o caminho


def edge_checker(value):
    
    x = clean_full_attr_df[clean_full_attr_df['Tuple']==value]
    if len(x)>1: ## means there its a duplicate
        min_val = x[x['Length']==x['Length'].min()]
        return (min_val.values[0][0], min_val.values[0][1], min_val.values[0][2], min_val.values[0][3])
    else:
        return (x.values[0][0], x.values[0][1], x.values[0][2], x.values[0][3])



def routes_attributes_list(path, cpus):

    edges_path = ((path[i], path[i+1]) for i in range(len(path)-1))
    path_attr_list = []
    pool = mp.Pool(cpus)
    sma = pool.map(edge_checker, edges_path)
    pool.close()
    pool.join()
    #print(len(sma))
    
    
    return sma




l_from_path = []
for i in range(len(paths['L from path'])):
    print(i)
    path = eval(paths['L from path'].iloc[i])
    l_from_path.append(routes_attributes_list(path,1))
    clear_output(True)




save = pd.DataFrame(zip(l_from_path), columns =['L from paths']) 
save.to_csv(city_name+'_fracoes.dat') ## escrever onde guardar aqui






