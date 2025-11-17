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


# First Algorithm (backtracking)

def backtracking_dfs(G, start):
    time_start = datetime.now()
    total_nodes = len(G.nodes())

    solution_path = []
    finished = False
    finish_node = None

    stack = deque()
    stack.append((start, [start], {start}))   # (node sekarang, path, visited set)

    while stack:
        node, path, visited = stack.pop()

        if len(path) == total_nodes:
            solution_path = path
            finished = True
            finish_node = path[-1]
            break

        for nb in G.neighbors(node):
            if nb not in visited:
                new_path = path + [nb]
                new_visited = visited | {nb}
                stack.append((nb, new_path, new_visited))

    time_finished = datetime.now() - time_start
    elapsed_s = time_finished.total_seconds()

    return solution_path, finished, finish_node, f"{elapsed_s:.6f} s ({elapsed_s*1000:.3f} ms)"


# Second Algorithm (backtracking + greedy)

def greedy_dfs(G, start):
    time_start = datetime.now()
    total_nodes = len(G.nodes())

    solution_path = []
    finished = False
    finish_node = None

    stack = deque()
    stack.append((start, [start], {start}))   # (node sekarang, path, visited set)

    while stack:
        node, path, visited = stack.pop()

        if len(path) == total_nodes:
            solution_path = path
            finished = True
            finish_node = path[-1]
            break

        neighbors = [nb for nb in G.neighbors(node) if nb not in visited]
        neighbors.sort(key=lambda x: G.degree(x), reverse=True)

        for nb in neighbors:
            if nb not in visited:
                new_path = path + [nb]
                new_visited = visited | {nb}
                stack.append((nb, new_path, new_visited))

    time_finished = datetime.now() - time_start
    elapsed_s = time_finished.total_seconds()

    return solution_path, finished, finish_node, f"{elapsed_s:.6f} s ({elapsed_s*1000:.3f} ms)"


# Third Algorithm (backtracking + greedy + forced move)

def forced_move_dfs(G, start):
    time_start = datetime.now()
    total_nodes = len(G.nodes())

    solution_path = []
    finished = False
    finish_node = None

    stack = deque()
    stack.append((start, [start], {start}))   # (node sekarang, path, visited set)

    while stack:
        node, path, visited = stack.pop()

        forced = True
        while forced:
            forced = False

            while True:
                neighbors = [nb for nb in G.neighbors(node) if nb not in visited]
                if len(neighbors) == 1:
                    node = neighbors[0]
                    path.append(node)
                    visited.add(node)
                    continue
                break

            neighbors = [nb for nb in G.neighbors(node) if nb not in visited]
            for nb in neighbors:
                forced_move = [nnb for nnb in G.neighbors(nb) if nnb not in visited and nnb != node]
                if len(forced_move) == 1:
                    node = nb
                    path.append(node)
                    visited.add(node)
                    forced = True
                    break

        if len(path) == total_nodes:
            solution_path = path
            finished = True
            finish_node = path[-1]
            break

        neighbors = [nb for nb in G.neighbors(node) if nb not in visited]
        neighbors.sort(key=lambda x: G.degree(x), reverse=True)

        for nb in neighbors:
            if nb not in visited:
                new_path = path + [nb]
                new_visited = visited | {nb}
                stack.append((nb, new_path, new_visited))

    time_finished = datetime.now() - time_start
    elapsed_s = time_finished.total_seconds()

    return solution_path, finished, finish_node, f"{elapsed_s:.6f} s ({elapsed_s*1000:.3f} ms)"


# Fourth Algorithm (backtracking + greedy + edge elimination)

def edge_elimination_dfs(G, start):
    time_start = datetime.now()
    total_nodes = len(G.nodes())

    solution_path = []
    finished = False
    finish_node = None

    nx.set_node_attributes(G, {n: G.degree(n) for n in G.nodes()}, "degree_value")
    nx.set_node_attributes(G, {n: 0 for n in G.nodes()}, "edge_value")

    remove_list = []
    append_list = []

    visited_edge = set()
    for node in list(G.nodes()):
        required_degree = 1 if (node == start or G.nodes[node]["degree_value"] == 1) else 2
        if G.nodes[node]["degree_value"] == required_degree:
            G.nodes[node]["edge_value"] = required_degree
            neighbors = list(G.neighbors(node))
            for nb in neighbors:
                visited_edge.add(tuple(sorted((node, nb))))
                required_degree = 1 if (nb == start or G.degree(nb) == 1) else 2
                if G.nodes[nb]["edge_value"] < required_degree:
                    G.nodes[nb]["edge_value"] += 1
                if G.nodes[nb]["edge_value"] == required_degree and G.nodes[nb]["degree_value"] > required_degree:
                    remove_list.append(nb)

    stack = deque()
    stack.append((start, [start], {start}, visited_edge, {None}))   # (node sekarang, path, visited node, visited edge, removed edge)

    while stack:
        node, path, visited_node, visited_edge, removed_edge = stack.pop()

        step = True
        while remove_list or step:
            step = False

            while True:
                neighbors = [nb for nb in G.neighbors(node) if tuple(sorted((node, nb))) in visited_edge and nb not in visited_node]
                if neighbors:
                    node = neighbors[0]
                    path.append(node)
                    visited_node.add(node)
                    continue
                break

            while remove_list:
                node_list = remove_list.pop()
                neighbors = list(G.neighbors(node_list))
                for nb in neighbors:
                    edge = tuple(sorted((node_list, nb)))
                    if edge not in visited_edge:
                        removed_edge.add(edge)
                        G.nodes[node_list]["degree_value"] -= 1

                        required_degree = 1 if (nb == start or G.nodes[nb]["degree_value"] == 1) else 2
                        if G.nodes[nb]["degree_value"] > required_degree:
                            G.nodes[nb]["degree_value"] -= 1

                        required_degree = 1 if (nb == start or G.nodes[nb]["degree_value"] == 1) else 2
                        if G.nodes[nb]["degree_value"] == required_degree:
                            append_list.append(nb)

            while append_list:
                node_list = append_list.pop()
                neighbors = list(G.neighbors(node_list))
                for nb in neighbors:
                    edge = tuple(sorted((node_list, nb)))
                    if edge not in removed_edge:
                        step = True
                        visited_edge.add(edge)
                        required_degree = 1 if (node_list == start or G.nodes[node_list]["degree_value"] == 1) else 2
                        G.nodes[node_list]["edge_value"] = required_degree
                        required_degree = 1 if (nb == start or G.nodes[nb]["degree_value"] == 1) else 2
                        if G.nodes[nb]["edge_value"] < required_degree:
                            G.nodes[nb]["edge_value"] += 1
                        if G.nodes[nb]["edge_value"] == required_degree and G.nodes[nb]["degree_value"] > required_degree:
                            remove_list.append(nb)

        if len(path) == total_nodes:
            solution_path = path
            finished = True
            finish_node = path[-1]
            break

        neighbors = [nb for nb in G.neighbors(node) if nb not in visited_node]
        neighbors.sort(key=lambda x: G.degree(x), reverse=True)

        for nb in neighbors:
            if nb not in visited_node and tuple(sorted((node, nb))) not in removed_edge:
                new_path = path + [nb]
                new_visited_node = visited_node | {nb}
                stack.append((nb, new_path, new_visited_node, visited_edge.copy(), removed_edge.copy()))

    time_finished = datetime.now() - time_start
    elapsed_s = time_finished.total_seconds()

    return solution_path, finished, finish_node, f"{elapsed_s:.6f} s ({elapsed_s*1000:.3f} ms)"


# Validation Algorithms

def tarjan_validation(G, start, visited_node=None, removed_edge=None):
    if visited_node:
        for node in list(G.nodes()):
            if node in visited_node and node != start:
                G.remove_node(node)

    if removed_edge:
        for edge in removed_edge:
            if edge and G.has_edge(edge[0], edge[1]):
                G.remove_edge(edge[0], edge[1])

    components = list(nx.connected_components(G))
    if len(components) > 1:
        return False

    mapping = {node: i for i, node in enumerate(G.nodes())}
    rev_map = {i: node for node, i in mapping.items()}

    start_idx = mapping[start]

    n = len(G)
    disc = [-1] * n
    low  = [-1] * n
    parent = [-1] * n
    time = 0

    stack_edges = []
    articulation_idx = set()

    disc[start_idx] = low[start_idx] = time
    time += 1

    root_children = 0
    dfs = [(start_idx, -1, 0)]   # (node, parent, next_neighbor_index)

    def validate_bcc(bcc_edges):
        nodes_inside = set()
        for x, y in bcc_edges:
            nodes_inside.add(x)
            nodes_inside.add(y)

        aps_in_bcc = nodes_inside.intersection(articulation_nodes)
        contains_start = (start in nodes_inside)

        if contains_start:
            if len(aps_in_bcc) > 1:
                return False
        else:
            if len(aps_in_bcc) > 2:
                return False

        return True

    while dfs:
        u, p, idx = dfs.pop()
        neighbors = list(G.neighbors(rev_map[u]))

        if idx < len(neighbors):
            v = mapping[neighbors[idx]]
            dfs.append((u, p, idx + 1))

            if disc[v] == -1:  # tree edge
                parent[v] = u

                if u == start_idx:
                    root_children += 1

                disc[v] = low[v] = time
                time += 1
                stack_edges.append((u, v))
                dfs.append((v, u, 0))

            elif v != p and disc[v] < disc[u]:  # back edge
                low[u] = min(low[u], disc[v])
                stack_edges.append((u, v))

        else:
            if p != -1:
                low[p] = min(low[p], low[u])

                if low[u] >= disc[p] and p != start_idx:
                    articulation_idx.add(p)

                    bcc_edges = []
                    while stack_edges:
                        a, b = stack_edges.pop()
                        bcc_edges.append((rev_map[a], rev_map[b]))
                        if (a == p and b == u) or (a == u and b == p):
                            break

                    articulation_nodes = {rev_map[x] for x in articulation_idx}
                    valid = validate_bcc(bcc_edges)
                    if not valid:
                        return False

    if stack_edges:
        bcc_edges = [(rev_map[a], rev_map[b]) for a, b in stack_edges]
        articulation_nodes = {rev_map[x] for x in articulation_idx}
        valid = validate_bcc(bcc_edges)
        if not valid:
            return False
    
    if root_children > 1:
        return False

    return True


# Fifth Algorithm (backtracking + greedy + forced move + validation)

def validation_forced_move_dfs(G, start):
    time_start = datetime.now()
    total_nodes = len(G.nodes())

    solution_path = []
    finished = False
    solution_finish_node = None

    for node in G.nodes():
        if G.degree(node) == 1 and node != start:
            if solution_finish_node:
                return solution_path, finished, solution_finish_node, "0.000000 s (0.000 ms)"
            solution_finish_node = node

    if solution_finish_node:
        n = total_nodes
        u, v = start
        nu, nv = solution_finish_node

        if n % 2 == 0:
            if not (u + v) % 2 != (nu + nv) % 2:
                solution_finish_node = None
                return solution_path, finished, solution_finish_node, "0.000000 s (0.000 ms)"
        else:
            if not (u + v) % 2 == (nu + nv) % 2:
                solution_finish_node = None
                return solution_path, finished, solution_finish_node, "0.000000 s (0.000 ms)"

    stack = deque()
    stack.append((start, [start], {start}, solution_finish_node))   # (node sekarang, path, visited set)

    while stack:
        node, path, visited, finish_node = stack.pop()

        valid_finish_node = True

        forced = True
        while forced:
            forced = False

            while True:
                neighbors = [nb for nb in G.neighbors(node) if nb not in visited]
                if len(neighbors) == 1:
                    node = neighbors[0]
                    path.append(node)
                    visited.add(node)
                    continue
                break

            neighbors = [nb for nb in G.neighbors(node) if nb not in visited]
            forced_moves = []
            for nb in neighbors:
                forced_move = [nnb for nnb in G.neighbors(nb) if nnb not in visited and nnb != node]
                if len(forced_move) == 1:
                    forced_moves.append(nb)
            
            if len(forced_moves) == 1:
                node = forced_moves[0]
                path.append(node)
                visited.add(node)
                forced = True
            elif len(forced_moves) > 1:
                if finish_node:
                    valid_finish_node = False
                    break

                n = len(G.nodes()) - len(visited) - 1
                u, v = node
                nu, nv = forced_moves[0]

                if n % 2 == 0:
                    if (u + v) % 2 != (nu + nv) % 2:
                        finish_node = forced_moves[0]
                    else:
                        valid_finish_node = False
                        break
                else:
                    if (u + v) % 2 == (nu + nv) % 2:
                        finish_node = forced_moves[0]
                    else:
                        valid_finish_node = False
                        break

                node = forced_moves[0]
                path.append(node)
                visited.add(node)
                forced = True

        if not valid_finish_node:
            continue

        if not tarjan_validation(G.copy(), node, visited_node=visited):
            continue

        if len(path) == total_nodes:
            solution_path = path
            finished = True
            solution_finish_node = path[-1]
            break

        neighbors = [nb for nb in G.neighbors(node) if nb not in visited]
        neighbors.sort(key=lambda x: G.degree(x), reverse=True)

        for nb in neighbors:
            if nb not in visited:
                new_path = path + [nb]
                new_visited = visited | {nb}
                stack.append((nb, new_path, new_visited, finish_node))

    time_finished = datetime.now() - time_start
    elapsed_s = time_finished.total_seconds()

    return solution_path, finished, solution_finish_node, f"{elapsed_s:.6f} s ({elapsed_s*1000:.3f} ms)"


# Sixth Algorithm (backtracking + greedy + edge elimination + validation)

def validation_edge_elimination_dfs(G, start):
    time_start = datetime.now()
    total_nodes = len(G.nodes())

    solution_path = []
    finished = False
    solution_finish_node = None

    for node in G.nodes():
        if G.degree(node) == 1 and node != start:
            if solution_finish_node:
                return solution_path, finished, solution_finish_node, "0.000000 s (0.000 ms)"
            solution_finish_node = node

    if solution_finish_node:
        n = total_nodes
        u, v = start
        nu, nv = solution_finish_node

        if n % 2 == 0:
            if not (u + v) % 2 != (nu + nv) % 2:
                solution_finish_node = None
                return solution_path, finished, solution_finish_node, "0.000000 s (0.000 ms)"
        else:
            if not (u + v) % 2 == (nu + nv) % 2:
                solution_finish_node = None
                return solution_path, finished, solution_finish_node, "0.000000 s (0.000 ms)"

    nx.set_node_attributes(G, {n: G.degree(n) for n in G.nodes()}, "degree_value")
    nx.set_node_attributes(G, {n: 0 for n in G.nodes()}, "edge_value")

    remove_list = []
    append_list = []

    visited_edge = set()
    for node in list(G.nodes()):
        required_degree = 1 if (node == start or G.nodes[node]["degree_value"] == 1) else 2
        if G.nodes[node]["degree_value"] == required_degree:
            G.nodes[node]["edge_value"] = required_degree
            neighbors = list(G.neighbors(node))
            for nb in neighbors:
                visited_edge.add(tuple(sorted((node, nb))))
                required_degree = 1 if (nb == start or G.degree(nb) == 1) else 2
                if G.nodes[nb]["edge_value"] < required_degree:
                    G.nodes[nb]["edge_value"] += 1
                if G.nodes[nb]["edge_value"] == required_degree and G.nodes[nb]["degree_value"] > required_degree:
                    remove_list.append(nb)

    stack = deque()
    stack.append((start, [start], {start}, visited_edge, {None}, solution_finish_node))   # (node sekarang, path, visited node, visited edge, removed edge, finish node)

    while stack:
        node, path, visited_node, visited_edge, removed_edge, finish_node = stack.pop()

        valid_finish_node = True

        step = True
        while (remove_list or step) and valid_finish_node:
            step = False

            while True:
                neighbors = [nb for nb in G.neighbors(node) if tuple(sorted((node, nb))) in visited_edge and nb not in visited_node]
                if neighbors:
                    node = neighbors[0]
                    path.append(node)
                    visited_node.add(node)
                    continue
                break

            while remove_list:
                node_list = remove_list.pop()
                neighbors = list(G.neighbors(node_list))
                for nb in neighbors:
                    edge = tuple(sorted((node_list, nb)))
                    if edge not in visited_edge:
                        removed_edge.add(edge)

                        required_degree = 1 if (node_list == start or G.nodes[node_list]["degree_value"] == 1) else 2
                        if G.nodes[node_list]["degree_value"] > required_degree:
                            G.nodes[node_list]["degree_value"] -= 1

                        required_degree = 1 if (nb == start or G.nodes[nb]["degree_value"] == 1) else 2
                        if G.nodes[nb]["degree_value"] > required_degree:
                            G.nodes[nb]["degree_value"] -= 1

                        if G.nodes[nb]["degree_value"] == 1 and nb != start:
                            if finish_node:
                                valid_finish_node = False
                                break
                                
                            n = len(G.nodes()) - len(visited_node) - 1
                            u, v = node
                            nu, nv = nb

                            if n % 2 == 0:
                                if (u + v) % 2 != (nu + nv) % 2:
                                    finish_node = nb
                                else:
                                    valid_finish_node = False
                                    break
                            else:
                                if (u + v) % 2 == (nu + nv) % 2:
                                    finish_node = nb
                                else:
                                    valid_finish_node = False
                                    break

                        required_degree = 1 if (nb == start or G.nodes[nb]["degree_value"] == 1) else 2
                        if G.nodes[nb]["degree_value"] == required_degree:
                            append_list.append(nb)

            while append_list and valid_finish_node:
                node_list = append_list.pop()
                neighbors = list(G.neighbors(node_list))
                for nb in neighbors:
                    edge = tuple(sorted((node_list, nb)))
                    if edge not in removed_edge:
                        step = True
                        visited_edge.add(edge)
                        required_degree = 1 if (node_list == start or G.nodes[node_list]["degree_value"] == 1) else 2
                        G.nodes[node_list]["edge_value"] = required_degree
                        required_degree = 1 if (nb == start or G.nodes[nb]["degree_value"] == 1) else 2
                        if G.nodes[nb]["edge_value"] < required_degree:
                            G.nodes[nb]["edge_value"] += 1
                        if G.nodes[nb]["edge_value"] == required_degree and G.nodes[nb]["degree_value"] > required_degree:
                            remove_list.append(nb)

        if not valid_finish_node:
            continue

        if not tarjan_validation(G.copy(), node, visited_node=visited_node, removed_edge=removed_edge):
            continue

        if len(path) == total_nodes:
            solution_path = path
            finished = True
            solution_finish_node = path[-1]
            break

        neighbors = [nb for nb in G.neighbors(node) if nb not in visited_node]
        neighbors.sort(key=lambda x: G.degree(x), reverse=True)

        for nb in neighbors:
            if nb not in visited_node and tuple(sorted((node, nb))) not in removed_edge:
                new_path = path + [nb]
                new_visited_node = visited_node | {nb}
                stack.append((nb, new_path, new_visited_node, visited_edge.copy(), removed_edge.copy(), finish_node))

    time_finished = datetime.now() - time_start
    elapsed_s = time_finished.total_seconds()

    return solution_path, finished, solution_finish_node, f"{elapsed_s:.6f} s ({elapsed_s*1000:.3f} ms)"
