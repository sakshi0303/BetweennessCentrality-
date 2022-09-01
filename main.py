import math
from multiprocessing import Process, Pool, Queue
from multiprocessing.pool import ThreadPool
import multiprocessing
import threading

import time
import pandas as pd
from heapq import heappush, heappop
from collections import defaultdict
import random

import networkx as nx
from nxBC import betweenness_centrality
from nxBC import _single_source_shortest_path_basic

from betweenessAndShortestPathCalculator import BetweenessCalculator

def mockNXGraph():
    G = nx.Graph()
    G.add_edges_from([
        (1,2),
        (1,4),
        (1,3),
        (2,4),
        (2,3),
        (2,8),
        (3,7),
        (4,6),
        (5,8),
        (6,5),
        (7,5)
    ])
    print(f"Total nodes being processed: {G.number_of_nodes()} and edges: {G.number_of_edges()}")   
    return G


#def denseGraph(n=5, degree=2):
def denseGraph(n,degree):
    G = nx.Graph()
    nodes = set()
    for i in range(0,n*degree):
        target = random.randint(1,n)
        source = random.randint(1,n)
        if target!=source:
            G.add_edge(source,target)
            nodes.add(source)
            nodes.add(target)
    for i in range(1,n):
        count = 0
        if G.has_node(i):
            for n in G.neighbors(i):
                count+=1
            continue
        G.add_edge(i, random.choice(list(nodes)))
    print(f"Total nodes being processed: {G.number_of_nodes()} and edges: {G.number_of_edges()}")   
    return G

def mockNXGraph2():
    G = nx.Graph()
    G.add_edges_from([
        (1,3),
        (1,5),
        (2,1),
        (2,5),
        (3,5),
        (4,5),
        (5,6),
    ])
    print(f"Total nodes being processed: {G.number_of_nodes()} and edges: {G.number_of_edges()}")   
    return G    

def loadNXGraph(path, limit):
    G = nx.Graph()
    lines = []
    with open(path) as f:
        lines = f.readlines()
    count = 0
    for line in lines:
        pre=line.split()
        for i in range(0, len(pre)):
            a=pre[0]
            b=pre[1]
            G.add_edge(a,b)      
        count += 1
        if count == limit:
            break
    print(f"Total nodes being processed: {G.number_of_nodes()} and edges: {G.number_of_edges()}")    
    return G

def buildAdjList(G):    
    res = defaultdict(set)
    for sub in list(G.edges):
        res[sub[0]].add(sub[1])
        res[sub[1]].add(sub[0])
    graph={}
    for k,v in res.items():
        v = [str(x) for x in v]
        graph[str(k)]=v
    return graph

# Uses naive BFS to find the shortest path from source to target node 
def Algo1(graph, betweenessCalculator):
    start_time = time.time()
    shortestPath = {}
    centrality = {}
    vertices = list(graph.keys())
    for (vertex, adjList) in graph.items():
      for nextVertex in vertices:
          if (vertex != nextVertex) and (nextVertex not in adjList) and (nextVertex, vertex) not in shortestPath:
              result = betweenessCalculator.BFS_sourceToTarget(graph, vertex, nextVertex, shortestPath)
              #print(f"Shortest path between {vertex} and {nextVertex} is {result}") 
    betweenessCalculator.standardnC2Approach(vertices, shortestPath, centrality)
    print("---Execution Time for intial BFS approach: %s seconds ---" % (time.time() - start_time))
    #print("\n Top 20% of nodes with highest betweenness_stress_centrality for BFS ",(sorted(centrality, key=centrality.get, reverse=True)[:math.ceil(.2*len(vertices))]))
    
    print(f"Betweenness_stress_centrality for intial BFS: {centrality}")
    print("------------------------------------------------------------------------")


def Algo2(graph, betweenessCalculator):
    start_time = time.time()
    shortestPath = {}
    centrality = {}
    vertices = list(graph.keys())

    for (vertex, adjList) in graph.items():
      for nextVertex in vertices:
          if (vertex != nextVertex) and (nextVertex not in adjList) and (nextVertex, vertex) not in shortestPath:
              result = betweenessCalculator.BiBFS_sourceToTarget(graph, vertex, nextVertex, shortestPath)
              #print(f"Shortest path between {vertex} and {nextVertex} is {result}") 

    betweenessCalculator.standardnC2Approach(vertices, shortestPath, centrality)
    print("---Execution Time for Bidirectional search: %s seconds ---" % (time.time() - start_time))
    print(" Top 20% of nodes with highest betweenness_stress_centrality for Bidirectional search: ",(sorted(centrality, key=centrality.get, reverse=True)[:math.ceil(.2*len(vertices))]))
    
    #print(f"Betweenness_stress_centrality for Bidirectional search: {centrality}")
    print("------------------------------------------------------------------------")

# (Sequential) Uses BFS to find shortest paths between source and all other nodes in 1 iteration

def accumulateCounts(S, P, sigma, s, betweennessCount, shortestPath):
    while S:
        target = S.pop()
        if ((s,target) in shortestPath) or ((target,s) in shortestPath):
          continue
        else:
          shortestPath[(s,target)] = True
        moveUp(P, target, s, betweennessCount)
    return betweennessCount

def moveUp(P, current, s, betweennessCount):
    if current == s:
        return
    for predecessor in P[current]:
        if predecessor != s:
            betweennessCount[predecessor] += 1
        moveUp(P, predecessor, s, betweennessCount)            
    

# Uses Multiple parallel processing to help speedup the algorithm when working with large datasets
def Algo4(graph, betweenessCalculator, processes=4):        
    # Initially used a queue but looks like it stops working after buffer gets full
    vertices = list(graph.keys())
    betweenness = dict.fromkeys(graph, 0.0)
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []     
    start = 0
    step = int(len(vertices)/processes)
    offset = step
    for i in range(processes):      
      p = Process(target=betweenessCalculator.BFS_parallel, args=(graph, i, vertices[start:offset], return_dict,))           
      start = offset+1
      offset = start+step          
      p.Daemon = True
      jobs.append(p)
    start_time = time.time()
    for p in jobs:
        try:
            p.start()
        except KeyboardInterrupt:
            p.terminate()
        except Exception as e:
            p.terminate()
        finally:
            p.join()    

    betweenness = dict.fromkeys(graph, 0.0)
    for key in betweenness:
        for i in range(processes):
            betweenness[key] += return_dict[i][key]
    #print(f"return_dict: {return_dict}")
    endtime = (time.time() - start_time)
    
    print(f"---Execution Time for parallel Multiprocessing with {processes} processes: {endtime} seconds ---")
    print(" Top 20% of nodes with highest betweenness_stress_centrality parallel Multiprocessing is: ",(sorted(betweenness, key=betweenness.get, reverse=True)[:math.ceil(.2*len(vertices))]))
    #print(f"Betweenness_stress_centrality for parallel Multiprocessing: {betweenness}")
    print("------------------------------------------------------------------------")
    
    

def Algo3(graph, betweenessCalculator):
    start_time = time.time()
    vertices = list(graph.keys())
    betweennessCount = dict.fromkeys(graph, 0.0)
    shortestPath = {}
    for v in vertices:
      S, P, sigma, D = betweenessCalculator.BFS_souceToVertices(graph, v)
      #print(f"v:{v}")
      #print(f"S:{S}")
      #print(f"P:{P}")
      #print(f"sigma:{sigma}")
      #print(f"D:{D}")
      #betweenness, delta = betweenessCalculator.ulrikBrandesApproach(betweenness, S, P, sigma, v)
      betweennessCount = accumulateCounts(S, P, sigma, v, betweennessCount, shortestPath)
    endtime = (time.time() - start_time)
    print(f"---Execution Time for BFS sequential processing with predecessor logic : {endtime} seconds ---")
    print(" Top 20% of nodes with highest betweenness_stress_centrality for BFS sequential processing with predecessor logic is: ",(sorted(betweennessCount, key=betweennessCount.get, reverse=True)[:math.ceil(.2*len(vertices))]))

    #print(f"betweenness_stress_centrality for {v} : {betweennessCount}")
    print("------------------------------------------------------------------------")  


def Algo5(graph, betweenessCalculator, threads=4):   
    vertices = list(graph.keys())
    betweenness = dict.fromkeys(graph, 0.0)
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []     
    start = 0
    step = int(len(vertices)/threads)
    offset = step
    for i in range(threads):      
      p = threading.Thread(target=betweenessCalculator.BFS_parallel, args=(graph, i, vertices[start:offset], return_dict,))                 
      start = offset+1
      offset = start+step          
      p.Daemon = True
      jobs.append(p)    
    start_time = time.time()
    for p in jobs:
        try:
            p.start()
        except KeyboardInterrupt:
            p.terminate()
        except Exception as e:
            p.terminate()
        finally:
            p.join()    

    betweenness = dict.fromkeys(graph, 0.0)
    for key in betweenness:
        for i in range(threads):
            betweenness[key] += return_dict[i][key]
    #print(f"return_dict: {return_dict}")
    endtime = (time.time() - start_time)
    
    print(f"---Execution Time for parallel Multithreading with {threads} threads: {endtime} seconds ---")
    print(" Top 20% of nodes with highest betweenness_stress_centrality parallel Multithreading is: ",(sorted(betweenness, key=betweenness.get, reverse=True)[:math.ceil(.2*len(vertices))]))
    #print(f"betweenness_stress_centrality parallel Multithreading is : {betweenness}")
    print("------------------------------------------------------------------------")
  
    

# NX library implementation
def Algo6(graph):
    vertices = graph.number_of_nodes()
    start_time = time.time()
    result = betweenness_centrality(graph, normalized = False, endpoints = False)
    #print("betweenness_stress_centrality",result)
    print("---Execution Time for NX library is: %s seconds ---" % (time.time() - start_time))
    print(" Top 20% of nodes with highest betweenness_stress_centrality in NX library betweenness_stress_centrality: ",(sorted(result, key=result.get, reverse=True)[:math.ceil(.2*vertices)]))
    #print("NX library betweenness_stress_centrality",result)
    print("------------------------------------------------------------------------")

    
# ---------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    betweenessCalculator = BetweenessCalculator()
    #G = mockNXGraph2() 
    G = denseGraph(100,2)
    #G = loadNXGraph("com-youtube.ungraph.txt", 10000)
    graph = buildAdjList(G)

    # BFS
    Algo1(graph, betweenessCalculator) 
    # ---------------------------------------------------------------------------------------------------

    # Bidirectional BFS for unweighted graph
    #Algo2(graph, betweenessCalculator)

    # ---------------------------------------------------------------------------------------------------
    
    # predecesor sequential algorithm
    Algo3(graph, betweenessCalculator)
    # ---------------------------------------------------------------------------------------------------
    
    # Multiprocessor Parallel Algorithm
    Algo4(graph, betweenessCalculator, 3)
    # ---------------------------------------------------------------------------------------------------
    
    # Multithread Parallel Algorithm
    Algo5(graph, betweenessCalculator, 3)
    # ---------------------------------------------------------------------------------------------------
    
    # NX library 
    Algo6(G)    
    # ---------------------------------------------------------------------------------------------------
