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
import time
from IPython.display import clear_output

ox.config(log_console=True, use_cache=True, cache_folder = '/home/diogo_mota/Dropbox/Diogo_Rede_simetria/Beta_Experiment/Network_Data') ## Path da pasta onde estao os jsons


city_name = 'Madrid'

cities = ox.geocode_to_gdf(['Spain, Madrid', 'Leganés', 'Getafe', 'Alcorcón', 'Pozuelo de Alarcón', 'Majadahonda', 'Las Rozas de Madrid',
                           'Torrelodones', 'Hoyo de Manzanares', 'Colmenar Viejo', 'Tres Cantos', 'Alcobendas', 'Rivas-Vaciamadrid',
                           'Coslada', 'San Fernando de Henares', 'Paracuellos de Jarama', 'San Sebastián de los Reyes'])

whole_polygon = cities.unary_union #unary union of both geometries
city_pol = cities['geometry'].iloc[0] #geometry of just the city

G_1 = ox.graph_from_polygon(city_pol, network_type='drive', simplify=True)
nodes_1, edges_1 = ox.graph_to_gdfs(G_1, nodes=True, edges=True)

G = ox.graph_from_polygon(whole_polygon, network_type='drive', simplify=True)
G_nx = nx.relabel.convert_node_labels_to_integers(G)
nodes, edges = ox.graph_to_gdfs(G_nx, nodes=True, edges=True)

weight = 'length'

nodes_list = nodes['osmid'].tolist()
nodes_to_use = [nodes_list.index(node) for node in list(G_1.nodes()) if node in nodes_list]

print('Nodes to use:',len(nodes_to_use))
n = 100000 ## Change number of pairs here
targets = np.random.choice(nodes_to_use, n, replace=True)
origins = np.random.choice(nodes_to_use, n, replace=True)


def shortest_path(orig, targ):
    try:
        
        l_from = ox.shortest_path(G_nx, targ, orig, weight=weight)
        l_to = ox.shortest_path(G_nx, orig, targ, weight=weight)
        l_from_length = nx.shortest_path_length(G_nx, targ, orig, weight=weight)
        l_to_length = nx.shortest_path_length(G_nx, orig, targ, weight=weight)
        
        return l_from, l_to, l_to_length, l_from_length
    
    except Exception:
        # for unsolvable routes (due to directed graph perimeter effects)
        return None


since = time.time()

cpus=3
params = ((orig, targ) for orig, targ in zip(origins, targets))
pool = mp.Pool(cpus)
sma = pool.starmap_async(shortest_path, params)
routes = sma.get()
pool.close()
pool.join()

final_time_elapsed = time.time() - since
print('Calculating routes took: {:.0f}m {:.0f}s'.format(final_time_elapsed // 60, final_time_elapsed % 60))

print('Before cleaning:', len(routes))

routes_clean = [val for val in routes if val!=None]
print('After cleaning:',len(routes_clean))


l_to_paths = [routes_clean[i][0] for i in range(len(routes_clean))]
l_from_paths = [routes_clean[i][1] for i in range(len(routes_clean))]
l_to_lengths = [routes_clean[i][2] for i in range(len(routes_clean))]
l_from_lengths = [routes_clean[i][3] for i in range(len(routes_clean))]


df_paths = pd.DataFrame(zip(l_to_paths, l_from_paths, l_to_lengths, l_from_lengths), columns = ['L to path', 'L from path', 'L to length', 
                                                                                        'L from length'])

df_paths.to_csv('/home/diogo_mota/Dropbox/Diogo_Rede_simetria/Beta_Experiment/'+city_name+'/'+city_name+'_100k_paths_no_shuffle.dat') ## onde guardar o dataframe
