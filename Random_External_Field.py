import networkx as nx
import numpy as np
import random

def add_edges(graph, node, generation, max_generation, order, p, H):
    if generation < max_generation:
        if node == (0, 0):
            children = [((generation + 1), i + 1) for i in range(order + 1)]
        else:
            base = (node[1] - 1) * order + 1
            children = [((generation + 1), base + i) for i in range(order)]
        for child in children:
            graph.add_node(child, spin=np.random.choice([-1, 1]), H=(H if random.random() <= p else 0))
            graph.add_edge(node, child)
            add_edges(graph, child, generation + 1, max_generation, order, p, H)

def calculate_energy(G, J=0.65):
    interaction_energy = 0
    field_energy = 0
    for node in G:
        S = G.nodes[node]['spin']
        H = G.nodes[node]['H']
        field_energy -= H * S
        for neighbor in G.neighbors(node):
            S_neighbor = G.nodes[neighbor]['spin']
            interaction_energy += -J * S * S_neighbor
    return (interaction_energy / 2) + field_energy

def total_magnetisation(G):
    return sum(nx.get_node_attributes(G, 'spin').values())

def magnetisation_per_spin(G):
    return total_magnetisation(G) / len(G)

def metropolis_step(G, beta, J=0.65):
    nodes = list(G.nodes())
    random.shuffle(nodes)
    for node in nodes:
        S = G.nodes[node]['spin']
        H = G.nodes[node]['H']
        neighbors = list(G.neighbors(node))
        if node != (0, 0):
            parent_generation = node[0] - 1
            parent_index = (node[1] - 1) // 2 + 1
            parent = (parent_generation, parent_index)
            if parent in G:
                neighbors.append(parent)
        neighbor_spins = sum(G.nodes[n]['spin'] for n in neighbors)
        dE = 2 * J * S * neighbor_spins + 2 * H * S  
        if dE < 0 or random.random() < np.exp(-beta * dE):
            G.nodes[node]['spin'] = -S

def simulate(G, T, steps, J=0.65):
    beta = 1.0 / T
    for step in range(steps):
        metropolis_step(G, beta, J)
    return G

def simulated_annealing(G, initial_T, final_T, steps, J=0.65):
    T = initial_T
    cooling_rate = (initial_T - final_T) / steps
    for step in range(steps):
        T -= cooling_rate
        beta = 1.0 / T
        metropolis_step(G, beta, J)
    return G

def calculate_node_statistics(G, max_generation):
    b = [0] * (max_generation + 1)
    r = [0] * (max_generation + 1)
    sb = [0] * (max_generation + 1)
    sr = [0] * (max_generation + 1)

    for generation in range(max_generation + 1):
        node_indices = sorted([node[1] for node in G.nodes() if node[0] == generation])
        max_red_streak = 0
        max_blue_streak = 0
        current_red_streak = 0
        current_blue_streak = 0

        for idx in node_indices:
            node = (generation, idx)
            spin = G.nodes[node]['spin']
            if spin == 1:
                current_red_streak += 1
                max_blue_streak = max(max_blue_streak, current_blue_streak)
                current_blue_streak = 0
            else:
                current_blue_streak += 1
                max_red_streak = max(max_red_streak, current_red_streak)
                current_red_streak = 0

        max_red_streak = max(max_red_streak, current_red_streak)
        max_blue_streak = max(max_blue_streak, current_blue_streak)

        r[generation] = sum(1 for node in G.nodes() if node[0] == generation and G.nodes[node]['spin'] == 1)
        b[generation] = sum(1 for node in G.nodes() if node[0] == generation and G.nodes[node]['spin'] == -1)
        sr[generation] = max_red_streak
        sb[generation] = max_blue_streak
    
    return b, r, sb, sr

# Setup and simulate
max_generation = 10
order = 2
H = 1 # External field 
p = 0.5 # Probability of a node having an external field
G = nx.Graph()
root = (0, 0)
G.add_node(root, spin=np.random.choice([-1, 1]), H=(H if random.random() < p else 0))
add_edges(G, root, 0, max_generation, order, p, H)

# Calculate initial energy and magnetisation
initial_energy = calculate_energy(G)
initial_total_magnetisation = total_magnetisation(G)
initial_magnetisation_per_spin = magnetisation_per_spin(G)
b, r, sb, sr = calculate_node_statistics(G, max_generation)
print(f'Initial Energy: {initial_energy}')
print(f'Initial Total Magnetisation: {initial_total_magnetisation}')
print(f'Initial Magnetisation per Spin: {initial_magnetisation_per_spin}')
print(f'Initial Blue Node Counts per Generation: {b}')
print(f'Initial Red Node Counts per Generation: {r}')
print(f'Initial Longest Blue Node Index per Generation: {sb}')
print(f'Initial Longest Red Node Index per Generation: {sr}')

# Simulated annealing   # Optimization process
initial_T = 100
final_T = 2.75
steps = 1000000
G = simulated_annealing(G, initial_T, final_T, steps)

# Calculate final energy and magnetisation
final_energy = calculate_energy(G)
final_total_magnetisation = total_magnetisation(G)
final_magnetisation_per_spin = magnetisation_per_spin(G)
b, r, sb, sr = calculate_node_statistics(G, max_generation)
print(f'Final Energy: {final_energy}')
print(f'Final Total Magnetisation: {final_total_magnetisation}')
print(f'Final Magnetisation per Spin: {final_magnetisation_per_spin}')
print(f'Final Blue Node Counts per Generation: {b}')
print(f'Final Red Node Counts per Generation: {r}')
print(f'Final Longest Blue Node Index per Generation: {sb}')
print(f'Final Longest Red Node Index per Generation: {sr}')
