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


ox.config(log_console=True, use_cache=True, cache_folder = '/Network_Data') 


def shortest_path(orig, targ):
    weight='length'
    try:
        
        l_from = ox.shortest_path(G_new, targ, orig, weight=weight)
        l_to = ox.shortest_path(G_new, orig, targ, weight=weight)
        l_from_length = nx.shortest_path_length(G_new, targ, orig, weight=weight)
        l_to_length = nx.shortest_path_length(G_new, orig, targ, weight=weight)
        
        return l_from, l_to, l_to_length, l_from_length
    
    except Exception:
        # for unsolvable routes (due to directed graph perimeter effects)
        return None

def edge_checker(value):
    #print(value)
    x = df_shuffle_network[df_shuffle_network['tuple']==value]
    #display(x)
    if len(x)>1: ## means there is a duplicate
        min_val = x[x['length']==x['length'].min()]
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
# In[2]:


## Load Data
city_name = 'Rome'

cities = ox.geocode_to_gdf(['Roma', 'Città del Vaticano', 'Fiumicino', 'Anguillara Sabazia',
                           'Trevignano Romano', 'Campagnano di Roma', 'Formello',
                           'Sacrofano', 'Riano', 'Monterotondo', 'Mentana', 'Fonte Nuova',
                           'Guidonia Montecelio', 'Tivoli', 'Poli', 'Palestrina',
                           'Gallicano nel Lazio', 'Monte Porzio Catone', 'Frascati',
                           'Grottaferrata', 'Ciampino', 'Marino', 'Castel Gandolfo',
                           'Ardea', 'Pomezia', 'Albano Laziale', 'Zagarolo', 'Montecompatri', 
                           'San Gregorio da Sassola', 'Castel San Pietro Romano', 'San Cesareo', 'Colonna'])

whole_polygon = cities.unary_union #unary union of both geometries
city_pol = cities['geometry'].iloc[0] #geometry of just the city

G_1 = ox.graph_from_polygon(city_pol, network_type='drive', simplify=True)
nodes_1, edges_1 = ox.graph_to_gdfs(G_1, nodes=True, edges=True)

G_nx = ox.graph_from_polygon(whole_polygon, network_type='drive', simplify=True)
nodes, edges = ox.graph_to_gdfs(G_nx, nodes=True, edges=True)


## Edge classfication

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


# In[4]:


round_length_list = [round(value,0) for value in length_list]
full_attr_df = pd.DataFrame(zip(edge_list, round_length_list, key_list, bool_list),
                           columns = ['Tuple', 'Length', 'K', 'Oneway'])
#full_attr_df


original_edges = [(full_attr_df['Tuple'].iloc[i], full_attr_df['Length'].iloc[i],
                  full_attr_df['K'].iloc[i], 
                  full_attr_df['Oneway'].iloc[i]) for i in range(len(full_attr_df))]


# # New algorithm

# In[8]:


index_list=[]
classification_list =[]
seen=[]
for val in original_edges:
    
    index = original_edges.index(val)
    inv_val = (val[0][::-1], val[1], val[2], val[3])
    
    if val not in seen:
        seen.append(inv_val)
        if val[3]== False and inv_val in original_edges:
            inv_index = original_edges.index(inv_val)
            index_list.append([index, inv_index])
            classification_list.append(False)

        else:
            index_list.append(index)
            classification_list.append(True)

# In[9]:

for number in range(10):


    shuffle_classification = rd.sample(classification_list, len(classification_list))

    print(len(shuffle_classification), len(index_list))


    # In[11]:


    shuffle_edges = []
    complete_shuffle_classification = []
    for j in range(len(index_list)):
        #print(index_list[j])
        if type(index_list[j])==int: ## originally one-way edge
            
            if shuffle_classification[j] == True:
                shuffle_edges.append(original_edges[index_list[j]][:3])
                complete_shuffle_classification.append(True)
            else:
                orig_edge = original_edges[index_list[j]][:3] ## everything but the classification
                shuffle_edges.append(orig_edge)
                complete_shuffle_classification.append(False)
                created_inverse = (orig_edge[0][::-1], orig_edge[1], orig_edge[2])
                shuffle_edges.append(created_inverse)
                complete_shuffle_classification.append(False)
                
            
        else: ## originally two-way egde
            if shuffle_classification[j]==True:
                index_keep = rd.sample(index_list[j], 1)[0]
                shuffle_edges.append(original_edges[index_keep][:3])
                complete_shuffle_classification.append(True)
            else:
                shuffle_edges.append(original_edges[index_list[j][0]][:3])
                complete_shuffle_classification.append(False)
                shuffle_edges.append(original_edges[index_list[j][1]][:3])
                complete_shuffle_classification.append(False)



    print(len(original_edges), len(shuffle_edges))

#    ## PLOTING

#    ## for the whole polygon
#    full_color_list =[0 for i in range(len(original_edges))]
#    for j in range(len(index_list)):
#        #print(index_list[j])
#        
#        if type(index_list[j])==int: ## originally one-way edge

#            if shuffle_classification[j] ==True:
#                full_color_list[index_list[j]]='firebrick'
#            else:
#                full_color_list[index_list[j]]='cornflowerblue'

#            

#            
#        else: ## originally two-way egde
#            index = index_list[j][0]
#            inv_index = index_list[j][1]

#            if shuffle_classification[j]==True:
#                full_color_list[index]='firebrick'
#                full_color_list[inv_index]='firebrick'
#            else:
#                full_color_list[index]='cornflowerblue'
#                full_color_list[inv_index]='cornflowerblue'


#    # In[22]:


#    print(len(full_color_list))


#    # In[23]:


#    original_edges_no_classif = [(full_attr_df['Tuple'].iloc[i], 
#        full_attr_df['K'].iloc[i]) for i in range(len(full_attr_df))]


#    # In[24]:


#    paris_only_list = [((edges_1['u'].iloc[j], edges_1['v'].iloc[j]), 
#        edges_1['key'].iloc[j]) for j in range(len(edges_1))]


#    # In[25]:


#    ## for paris
#    color_list =[0 for i in range(len(original_edges))]
#    for j in range(len(index_list)):
#        #print(index_list[j])
#        
#        if type(index_list[j])==int: ## originally one-way edge
#            if original_edges_no_classif[index_list[j]] in paris_only_list:

#                if shuffle_classification[j] ==True:
#                    color_list[index_list[j]]='firebrick'
#                else:
#                    color_list[index_list[j]]='cornflowerblue'
#            else:
#                color_list[index_list[j]] = None
#                
#            
#        else: ## originally two-way egde
#            index = index_list[j][0]
#            inv_index = index_list[j][1]
#            if original_edges_no_classif[index] in paris_only_list and original_edges_no_classif[inv_index] in paris_only_list:

#                if shuffle_classification[j]==True:
#                    color_list[index]='firebrick'
#                    color_list[inv_index]='firebrick'
#                else:
#                    color_list[index]='cornflowerblue'
#                    color_list[inv_index]='cornflowerblue'
#            else:
#                color_list[index] = None
#                color_list[inv_index] = None


#    # In[26]:


#    print(len(color_list))


#    # In[27]:


#    color_clean = [val for val in color_list if val != None]
#    print(len(color_clean))



#    fig, ax = ox.plot_graph(G_nx, node_size=0, edge_color=full_color_list, edge_linewidth=3, 
#        edge_alpha=0.7,bgcolor='white', show=False,figsize = (60, 40))
#    fig.set_frameon(True)
#    fig.tight_layout()
##    fig.savefig('/Shuffle_maps/Full_Paris_shuffled_new_algorithm_'+str(number)+'.png')

#    fig.savefig('/home/hygor/Dropbox/Pesquisa/Projetos/Diogo_Rede_simetria/Shuffle_maps/Full_'+city_name+'_shuffled_new_algorithm_'+str(number)+'_teste.png')

#    fig, ax = ox.plot_graph(G_1, node_size=0, edge_color=color_clean, 
#        edge_linewidth=3, edge_alpha=0.7,bgcolor='white', show=False,figsize = (30, 20))
#    fig.set_frameon(True)
#    fig.tight_layout()

#    fig.savefig('/Shuffle_maps/'+city_name+'_shuffled_new_algorithm_'+str(number)+'_teste.png')

    # Path calculations:

    

    shuffle_edges_list = [(val[0][0], val[0][1]) for val in shuffle_edges] ## no need for the key
    shuffle_length_list = [val[1] for val in shuffle_edges]




    print(len(shuffle_edges_list))




    attrs = {}
    for tup, length in zip(shuffle_edges_list, shuffle_length_list):
        attrs[tup] = {'length':length}




    ## Create the network with the new edges
    G_new = nx.MultiDiGraph(directed=True)
    G_new.add_nodes_from(G_nx.nodes())




    for val, length in zip(shuffle_edges_list, shuffle_length_list):
        u = val[0]
        v = val[1]
        G_new.add_edge(u,v, length=length)




    print(len(G_new.edges))




    lista = []
    for edge in G_new.edges:
        u = edge[0]
        v = edge[1]
        k = edge[2]
        lista.append(((u,v), G_new[u][v][k]['length'], k))




    nodes_to_use = [node for node in G_1.nodes() if node in G_nx.nodes()]

    print('Nodes to use:',len(nodes_to_use))
    n = 10000 ## Change number of pairs here
    targets = np.random.choice(nodes_to_use, n, replace=True)
    origins = np.random.choice(nodes_to_use, n, replace=True)


    since = time.time()

    cpus=2## mudar o numero de cores aqui
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


    df_paths.to_csv('/Shuffle_data/'+city_name+'_teste/'+city_name+'_100k_paths_shuffle_'+str(number)+'.dat') ## onde guardar o dataframe


    network_data = [(val[0], val[1], val[2],
                     value) for val, value in zip(lista, complete_shuffle_classification)]




    df_shuffle_network = pd.DataFrame(network_data, columns = ['tuple', 'length', 'key', 'one way'])




    df_shuffle_network.to_csv('/Shuffle_data/'+city_name+'_teste/'+city_name+'_shuffle_network_'+str(number)+'.dat')

    # # Fraction calculation:




    l_from_path_info = []
    for i in range(len(df_paths['L from path'])):
        print(i)
        path = df_paths['L from path'].iloc[i]
        l_from_path_info.append(routes_attributes_list(path,2))
        clear_output(True)



    save = pd.DataFrame(zip(l_from_path_info), columns =['L from paths']) 

    save.to_csv('/Shuffle_data/'+city_name+'_teste/'+city_name+'_fractions_shuffle_'+str(number)+'.dat')



