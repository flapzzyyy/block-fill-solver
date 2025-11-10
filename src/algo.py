import numpy as np
import networkx as nx
from datetime import datetime
from collections import deque

# Convert Matrix to Graph
def get_graph_from_binary_matrix(mat):
    arr = np.array(mat, dtype=int)
    rows, cols = arr.shape
    
    G = nx.Graph()
    start = None

    for r in range(rows):
        for c in range(cols):
            if arr[r, c] == 1:
                G.add_node((r, c))
            elif arr[r, c] == 2:
                G.add_node((r, c))
                start = (r, c)

    neighbor = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r, c in list(G.nodes()):
        for dr, dc in neighbor:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if arr[nr, nc] == 1 or arr[nr, nc] == 2:
                    G.add_edge((r, c), (nr, nc))

    return G, start


# --- Secondary Algorithm ---


def tarjan_articulation_points(G):
    time = 0
    disc = {}
    low = {}
    parent = {}
    ap = set()

    def tarjan_dfs(u):
        nonlocal time
        disc[u] = low[u] = time
        time += 1
        children = 0
        for v in G.neighbors(u):
            if v not in disc:
                parent[v] = u
                children += 1
                tarjan_dfs(v)
                low[u] = min(low[u], low[v])
                if parent.get(u, None) is None and children > 1:
                    ap.add(u)
                if parent.get(u, None) is not None and low[v] >= disc[u]:
                    ap.add(u)
            elif v != parent.get(u, None):
                low[u] = min(low[u], disc[v])

    for node in G.nodes():
        if node not in disc:
            parent[node] = None
            tarjan_dfs(node)

    return ap


def count_components(G, start):
    visited = set()
    visited.add(start)
    components = 0

    for node in G.nodes():
        if node not in visited:
            components += 1
            
            queue = deque([node])
            visited.add(node)

            while queue:
                u = queue.popleft()
                for v in G.neighbors(u):
                    if v not in visited:
                        visited.add(v)
                        queue.append(v)

    return components


def count_d1_nodes(G, start):
    d1_nodes = 0

    for node in G.nodes():
        if G.degree(node) == 1:
            d1_nodes += 1

    return d1_nodes - (G.degree(start) == 1)


def valid_init(G, start):
    ap_init = tarjan_articulation_points(G)

    if start in ap_init or count_components(G, start) > 1 or count_d1_nodes(G, start) > 1:
        return False
    
    for ap in tarjan_articulation_points(G):
        if count_components(G, ap) > 2:
            return False
        
    return True


# --- Main Algorithm ---


def optimized_dfs(G, start):
    time_start = datetime.now()    

    total_nodes = len(G.nodes())
        
    def main_dfs(G, start, current_path=None):
        if current_path == None:
            current_path = []

        current = start
        current_path.append(current)

        if count_components(G, current) > 1:
            return current_path, False, None

        moved = True
        while moved:
            moved = False

            d1_nodes = True
            while d1_nodes:
                d1_nodes = False
                if G.degree(current) == 1:
                    next = [nb for nb in G.neighbors(current)][0]
                    G.remove_node(current)
                    current = next
                    current_path.append(current)
                    d1_nodes = True

            d2_neighbors = [nb for nb in G.neighbors(current) if G.degree(nb) == 2]
            if d2_neighbors:
                next = d2_neighbors[0]
                G.remove_node(current)
                current = next
                if len(d2_neighbors) > 1:
                    if count_d1_nodes(G, current) > 1:
                        return current_path, False, None
                current_path.append(current)
                moved = True

        if count_components(G, current) > 1:
            return current_path, False, None

        if len(current_path) == total_nodes:
            return current_path, True, current_path[-1]

        finished = False
        finish_node = None
        
        current_neighbors = list(G.neighbors(current))
        G.remove_node(current)

        for nb in current_neighbors:
            G_copy = G.copy()
            path_copy = current_path.copy()
            path_copy, finished, finish_node = main_dfs(G_copy, nb, path_copy)
            if finished:
                current_path = path_copy
                break

        return current_path, finished, finish_node

    path = []
    finished = False
    finish_node = None

    if valid_init(G, start):
        path, finished, finish_node = main_dfs(G, start)
        path = [tuple(p) for p in path]

    time_finished = datetime.now() - time_start 
    elapsed_s = time_finished.total_seconds()

    return path, finished, finish_node, f"{elapsed_s:.6f} s ({elapsed_s*1000:.3f} ms)"


def backtracking_dfs(G, start):
    time_start = datetime.now()

    visited = set()
    path = []

    def dfs(node):
        visited.add(node)
        path.append(node)

        if len(path) == len(G.nodes()):
            return True

        for nb in G.neighbors(node):
            if nb not in visited:
                if dfs(nb):
                    return True

        visited.remove(node)
        path.pop()
        return False
    
    path = []
    finish_node = None
    finished = dfs(start)

    if finished:
        finish_node = path[-1]
        path = [tuple(p) for p in path]

    time_finished = datetime.now() - time_start
    elapsed_s = time_finished.total_seconds()

    return path, finished, finish_node, f"{elapsed_s:.6f} s ({elapsed_s*1000:.3f} ms)"


def greedy_dfs(G, start):
    time_start = datetime.now()

    visited = set()
    path = []

    def dfs(node):
        visited.add(node)
        path.append(node)

        if len(path) == len(G.nodes()):
            return True

        neighbors = [nb for nb in G.neighbors(node) if nb not in visited]
        neighbors.sort(key=lambda x: G.degree(x))

        for nb in neighbors:
            if dfs(nb):
                return True

        visited.remove(node)
        path.pop()
        return False
    
    path = []
    finish_node = None
    finished = dfs(start)
    
    if finished:
        finish_node = path[-1]
        path = [tuple(p) for p in path]

    time_finished = datetime.now() - time_start
    elapsed_s = time_finished.total_seconds()

    return path, finished, finish_node, f"{elapsed_s:.6f} s ({elapsed_s*1000:.3f} ms)"


def edge_elimination(G, start):
    time_start = datetime.now()

    total_nodes = len(G.nodes())
    nx.set_node_attributes(G, 0, "visited_edge")

    def main_elimination(G, start, visited_edge=None):
        if visited_edge is None:
            visited_edge = set()
        
        eliminated = True
        while eliminated:
            eliminated = False

            for node in list(G.nodes()):
                if G.nodes[node]["visited_edge"] == 2:
                    continue

                if (node == start or G.degree(node) == 1) and G.nodes[node]["visited_edge"] == 1:
                    continue

                required_degree = 1 if (node == start or G.degree(node) == 1) else 2

                neighbors = list(G.neighbors(node))
                if len(neighbors) == required_degree:
                    G.nodes[node]["visited_edge"] = required_degree
                    for nb in neighbors:
                        if (nb == start or G.degree(nb) == 1) and G.nodes[nb]["visited_edge"] == 1:
                            continue
                            
                        edge = tuple(sorted((node, nb)))
                        visited_edge.add(edge)
                        if G.nodes[nb]["visited_edge"] < 2:
                            G.nodes[nb]["visited_edge"] += 1
                        
            for node in list(G.nodes()):
                should_remove = False
                
                if G.nodes[node]["visited_edge"] == 2 and G.degree(node) > 2:
                    should_remove = True
                elif node == start and G.nodes[node]["visited_edge"] == 1 and G.degree(node) > 1:
                    should_remove = True
                
                if should_remove:
                    edges_to_remove = []
                    for u, v in G.edges(node):
                        if tuple(sorted((u, v))) not in visited_edge:
                            edges_to_remove.append((u, v))
                    
                    selected_nodes = None
                    if not edges_to_remove and G.degree(node) > 2:
                        visited = set()

                        def dfs(u):
                            if u == start:
                                return True
                            visited.add(u)
                            for v in G.neighbors(u):
                                if v not in visited:
                                    if dfs(v):
                                        return True
                            return False

                        for nb in G.neighbors(node):
                            if nb not in visited:
                                selected_nodes = nb
                                if not dfs(nb):
                                    break

                        edges_to_remove.append((node, selected_nodes))
                        edge = tuple(sorted((node, nb)))
                        visited_edge.remove(edge)

                    for u, v in edges_to_remove:
                        if G.has_edge(u, v):
                            G.remove_edge(u, v)
                            eliminated = True

        if count_components(G, start) > 1 or count_d1_nodes(G, start) > 1:
            return False, G
            
        if len(visited_edge) == total_nodes - 1:
            return True, G
        
        finished = False
        while not finished:
            G_copy = G.copy()
            
            target_node = []
            queue = deque([start])
            visited_bfs = {start}
            parent_map = {start: None}

            if G.nodes[start]["visited_edge"] == 0:
                target_node.append(start)
            
            while queue:
                current = queue.popleft()
                
                for nb in G.neighbors(current):
                    if nb not in visited_bfs:
                        visited_bfs.add(nb)
                        parent_map[nb] = current
                        
                        if G.nodes[nb]["visited_edge"] == 1:
                            trace_node = nb
                            if target_node:
                                while parent_map[trace_node] != target_node[0]:
                                    trace_node = parent_map[trace_node]
                            target_node.append(trace_node)

                        if len(target_node) == 2:
                            break

                        queue.append(nb)
            
                if len(target_node) == 2:
                    break
            
            if len(target_node) < 2:
                break

            temp_visited_edge = visited_edge.copy()
            temp_visited_edge.add(tuple(sorted((target_node[0], target_node[1]))))

            for node in target_node:
                if (node == start or G.degree(node) == 1) and G.nodes[node]["visited_edge"] == 1:
                    continue
                if G_copy.nodes[node]["visited_edge"] < 2:
                    G_copy.nodes[node]["visited_edge"] += 1

            finished, G_copy = main_elimination(G_copy, start, temp_visited_edge)

            if finished:
                return True, G_copy
            else:
                if G.has_edge(target_node[0], target_node[1]):
                    G.remove_edge(target_node[0], target_node[1])

        return finished, G
    
    path = []
    finished = False
    finish_node = None

    if valid_init(G, start):
        finished, final_graph = main_elimination(G, start)
        
        if finished:
            current = start
            visited_path = {current}
            path = [current]
            
            while len(path) < total_nodes:
                neighbors = [nb for nb in final_graph.neighbors(current) if nb not in visited_path]
                if not neighbors:
                    break
                current = neighbors[0]
                visited_path.add(current)
                path.append(current)
            
            finish_node = path[-1] if path else None

    time_finished = datetime.now() - time_start 
    elapsed_s = time_finished.total_seconds()

    return path, finished, finish_node, f"{elapsed_s:.6f} s ({elapsed_s*1000:.3f} ms)"
