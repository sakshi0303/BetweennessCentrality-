from multiprocessing import Process, Pool, Queue
from multiprocessing.pool import ThreadPool
import sys
from heapq import heappush, heappop
from collections import deque

class BetweenessCalculator:

    def __init__(self):
        pass

    def BFS_parallel(self, G, i, vertices, return_dict):
        results = []
        betweenness = dict.fromkeys(G, 0.0)
        for v in vertices:
            S, P, sigma, D = self.BFS_souceToVertices(G, v)
            betweenness = self.accumulateCounts(S, P, sigma, v, betweenness, return_dict)
            #betweenness, delta =self.ulrikBrandesApproach(betweenness, S, P, sigma, v)
        return_dict[i] = betweenness    

    def accumulateCounts(self,S, P, sigma, s, betweennessCount, shortestPath):
        while S:
            target = S.pop()
            if ((s,target) in shortestPath) or ((target,s) in shortestPath):
                continue
            else:
                shortestPath[(s,target)] = True
            self.moveUp(P, target, s, betweennessCount)
        return betweennessCount

    def moveUp(self,P, current, s, betweennessCount):
        if current == s:
            return
        for predecessor in P[current]:
            if predecessor != s:
                betweennessCount[predecessor] += 1
            self.moveUp(P, predecessor, s, betweennessCount)            
    
    # Finds all the shortest paths between source and all the other vertices
    def BFS_souceToVertices(self, G, source):
        S = []
        P = {} # predecessors
        for v in G:
            P[v] = []
        sigma = dict.fromkeys(G, 0.0)  # sigma[v]=0 for v in G
        D = {} # distance map D[vertex] = distance
        sigma[source] = 1.0 
        D[source] = 0 # shortest path from s to target node
        Q = deque([source])
        while Q:  # use BFS to find shortest paths
            v = Q.popleft()
            S.append(v)
            Dv = D[v]
            sigmav = sigma[v]
            for w in G[v]:
                if w not in D:
                    Q.append(w)
                    D[w] = Dv + 1
                if D[w] == Dv + 1:  # this is a shortest path, count paths
                    sigma[w] += sigmav
                    P[w].append(v)  # predecessors
        return S, P, sigma, D


    # http://snap.stanford.edu/class/cs224w-readings/brandes01centrality.pdf
    # Introduces the concept of dependency where each source node contributes to dependency on a vertext
    # This value is calculated using the formula: delta[v] = sigma[v]*(1+delta[w])/sigma[w]
    # w is the target node
    # v is the node that passes through source and target
    # Higher the dependence of a source node to a vertex, the more is its importance
    #
    # sigma[v] is the number of shortest paths from source to vertex
    # sigma[w] is the number of shortest paths from source to target
    # delta[w] is the dependence of source on target node
    def ulrikBrandesApproach(self, betweenness, S, P, sigma, s):
        delta = dict.fromkeys(S, 0)
        while S:
            w = S.pop()
            coeff = (1 + delta[w]) / sigma[w]
            for v in P[w]:
                delta[v] += sigma[v] * coeff
                #print(f"delta[v]: {delta[v]}")
            if w != s:
                #print(f"delta[w]: {delta[w]}")              
                betweenness[w] += delta[w]
                #print(f"betweenness[w]: {betweenness[w]}")                
        return betweenness, delta

    # count the number of shortest paths that pass through the vertex and divide by nC2
    def standardnC2Approach(self, vertices, shortestPath, centrality):
        for key, values in shortestPath.items():
            count = len(values[0])
            minPathId = 0
            for i, path in enumerate(values):
                if len(path)>count :
                    del(shortestPath[key][i])
                elif count>len(path):
                    count = len(path)
                    del(shortestPath[key][minPathId])
                    minPathId = i
        shortPath = str(list(shortestPath.values()))
        for vertex in vertices:
            if vertex in shortPath:
                centrality[vertex] = shortPath.count(vertex)

    def shortestPathDjikstra(self,G, s, weight):
        weight = self._weight_function(G, weight)
        S = []
        P = {}
        for v in G:
            P[v] = []
        sigma = dict.fromkeys(G, 0.0)  # sigma[v]=0 for v in G
        D = {}
        sigma[s] = 1.0
        push = heappush
        pop = heappop
        seen = {s: 0}
        c = count()
        Q = []  # use Q as heap with (distance,node id) tuples
        push(Q, (0, next(c), s, s))
        while Q:
            (dist, _, pred, v) = pop(Q)
            if v in D:
                continue  # already searched this node.
            sigma[v] += sigma[pred]  # count paths
            S.append(v)
            D[v] = dist
            for w, edgedata in G[v].items():
                vw_dist = dist + weight(v, w, edgedata)
                if w not in D and (w not in seen or vw_dist < seen[w]):
                    seen[w] = vw_dist
                    push(Q, (vw_dist, next(c), v, w))
                    sigma[w] = 0.0
                    P[w] = [v]
                elif vw_dist == seen[w]:  # handle equal paths
                    sigma[w] += sigma[v]
                    P[w].append(v)
        return S, P, sigma, D

    def _weight_function(G, weight):        
        if G.is_multigraph():
            return lambda u, v, d: min(attr.get(weight, 1) for attr in d.values())
        return lambda u, v, data: data.get(weight, 1)


    def BiBFS_sourceToTarget(self, graph, start, goal, shortestPath):
        if start == goal:
            return [start]
        active_vertices_path_dict = {start: [start], goal: [goal]}
        paths = []
        maxlen = sys.maxsize
        inactive_vertices = set()

        while len(active_vertices_path_dict) > 0:
            active_vertices = list(active_vertices_path_dict.keys())
            for vertex in active_vertices:
                current_path = active_vertices_path_dict[vertex]
                # Record whether we started at start or goal.
                origin = current_path[0]
                # Check for new neighbours.
                current_neighbours = set(graph[vertex]) - inactive_vertices
                # Check if our neighbours hit an active vertex
                if len(current_neighbours.intersection(active_vertices)) > 0:
                    for meeting_vertex in current_neighbours.intersection(active_vertices):
                        # Check the two paths didn't start at same place. If not, then we've got a path from start to goal.
                        if origin != active_vertices_path_dict[meeting_vertex][0]:
                            # Reverse one of the paths.
                            active_vertices_path_dict[meeting_vertex].reverse()
                            # return the combined results
                            possible_path = active_vertices_path_dict[vertex] + active_vertices_path_dict[meeting_vertex]
                            if len(possible_path) <= maxlen:
                            	paths.append(possible_path[1:-1])
                            	maxlen = len(possible_path)

                # No hits, so check for new neighbours to extend our paths.
                if len(set(current_neighbours) - inactive_vertices - set(active_vertices))  == 0:
                    # If none, then remove the current path and record the endpoint as inactive.
                    active_vertices_path_dict.pop(vertex, None)
                    inactive_vertices.add(vertex)
                else:
                    # Otherwise extend the paths, remove the previous one and update the inactive vertices.
                    for neighbour_vertex in current_neighbours - inactive_vertices - set(active_vertices):
                        active_vertices_path_dict[neighbour_vertex] = current_path + [neighbour_vertex]
                        active_vertices.append(neighbour_vertex)
                    active_vertices_path_dict.pop(vertex, None)
                    inactive_vertices.add(vertex)

        shortestPath[(start, goal)] = (paths)
        return paths

    def BFS_sourceToTarget(self, graph, start, goal, shortestPath):
        explored = []
        # Queue for traversing the graph in the BFS
        queue = [[start]]        
        # If the desired node is reached
        if start == goal:
            return        
        # Loop to traverse the graph with the help of the queue
        while queue:
            path = queue.pop(0)
            if path[-1] != goal:
                node = path[-1]
            else:
                # get the list after removing the last item [1,2,3,4] becomes [1,2,3]
                path = path[0:-1]
                # get the last item in the list -> 3
                node = path[-1]
            # Condition to check if the current node is not visited
            if node not in explored:
                neighbours = graph[node]
                # Loop to iterate over the neighbours of the node
                for neighbour in neighbours:
                    new_path = list(path)
                    new_path.append(neighbour)
                    queue.append(new_path)
                    # Condition to check if the neighbour node is the goal
                    if neighbour == goal and new_path[0]==start and len(new_path) == 2:
                        return
                    if neighbour == goal and new_path[0]==start and len(new_path) > 2 :
                        if (start, goal) in shortestPath:
                            shortestPath[(start, goal)].append(new_path[1:-1])
                        else:
                            shortestPath[(start, goal)] = [list(new_path[1:-1])]
                        break
                explored.append(node)
        return