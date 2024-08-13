import networkx as nx
import numpy as np
import random

def add_edges(graph, node, generation, max_generation, g, p):
    if generation < max_generation:
        num_children = generation + 2
        children = [((generation + 1), (node[1] - 1) * num_children + i + 1) for i in range(num_children)]
        for child in children:
            if generation + 1 == g:  
                graph.add_node(child, spin=-1, fixed=True, boundary=True)
            else:
                if random.random() < p:
                    spin = 1
                    fixed = True
                else:
                    spin = np.random.choice([-1, 1])
                    fixed = False
                graph.add_node(child, spin=spin, fixed=fixed, boundary=False)
            graph.add_edge(node, child)
            add_edges(graph, child, generation + 1, max_generation, g, p)

def calculate_energy(G, J=0.65):
    interaction_energy = 0
    for node in G:
        if not G.nodes[node].get('boundary', False):
            S = G.nodes[node]['spin']
            for neighbor in G.neighbors(node):
                S_neighbor = G.nodes[neighbor]['spin']
                interaction_energy += -J * S * S_neighbor
    return interaction_energy / 2

def total_magnetisation(G):
    return sum(G.nodes[node]['spin'] for node in G if not G.nodes[node].get('boundary', False))

def magnetisation_per_spin(G):
    count = sum(1 for node in G if not G.nodes[node].get('boundary', False))
    return total_magnetisation(G) / count if count > 0 else 0

def metropolis_step(G, beta, J=0.65):
    nodes = list(G.nodes())
    random.shuffle(nodes)
    for node in nodes:
        if G.nodes[node].get('fixed', False):
            continue
        S = G.nodes[node]['spin']
        neighbors = list(G.neighbors(node))
        if node != (0, 0):
            parent_generation = node[0] - 1
            parent_index = (node[1] - 1) // (node[0]) + 1
            parent = (parent_generation, parent_index)
            if parent in G:
                neighbors.append(parent)
        neighbor_spins = sum(G.nodes[n]['spin'] for n in neighbors)
        dE = 2 * J * S * neighbor_spins
        if dE < 0 or random.random() < np.exp(-dE * beta):
            G.nodes[node]['spin'] *= -1
    return G

def simulate(G, T, steps, J=0.65):
    beta = 1.0 / T
    for _ in range(steps):
        G = metropolis_step(G, beta, J)
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

        r[generation] = sum(1 for node in G.nodes() if node[0] == generation and G.nodes[node]['spin'] == 1 and not G.nodes[node].get('boundary', False))
        b[generation] = sum(1 for node in G.nodes() if node[0] == generation and G.nodes[node]['spin'] == -1 and not G.nodes[node].get('boundary', False))
        sr[generation] = max_red_streak
        sb[generation] = max_blue_streak
    
    return b, r, sb, sr

def initialize_tree(order, max_generation, g, p):
    G = nx.Graph()
    root = (0, 0)
    if random.random() < p:
        G.add_node(root, spin=1, fixed=True, boundary=False)
    else:
        G.add_node(root, spin=np.random.choice([-1, 1]), fixed=False, boundary=False)
    add_edges(G, root, 0, max_generation, g, p)
    return G

def count_spins(G):
    count_plus1 = sum(1 for node in G if G.nodes[node]['spin'] == 1 and not G.nodes[node].get('boundary', False))
    count_minus1 = sum(1 for node in G if G.nodes[node]['spin'] == -1 and not G.nodes[node].get('boundary', False))
    return count_plus1, count_minus1

def run_simulation(order, max_generation, p, steps):
    g = max_generation + 1
    G = initialize_tree(order, max_generation, g, p)
    
    initial_plus1, initial_minus1 = count_spins(G)
    initial_energy = calculate_energy(G)
    initial_total_magnetisation = total_magnetisation(G)
    initial_magnetisation_per_spin = magnetisation_per_spin(G)
    b, r, sb, sr = calculate_node_statistics(G, max_generation)
    
    print(f"Initial +1 spins: {initial_plus1}")
    print(f"Initial -1 spins: {initial_minus1}")
    print(f'Initial Energy: {initial_energy}')
    print(f'Initial Total Magnetisation: {initial_total_magnetisation}')
    print(f'Initial Magnetisation per Spin: {initial_magnetisation_per_spin}')
    print(f'Initial Blue Node Counts per Generation: {b}')
    print(f'Initial Red Node Counts per Generation: {r}')
    print(f'Initial Longest Blue Node Index per Generation: {sb}')
    print(f'Initial Longest Red Node Index per Generation: {sr}')
    
    fixed_plus1 = sum(1 for node in G if G.nodes[node]['spin'] == 1 and G.nodes[node].get('fixed', False))
    print(f"Initial fixed +1 spins: {fixed_plus1}")
    
    T = 1
    G = simulate(G, T, steps)
    
    final_plus1, final_minus1 = count_spins(G)
    final_energy = calculate_energy(G)
    final_total_magnetisation = total_magnetisation(G)
    final_magnetisation_per_spin = magnetisation_per_spin(G)
    b, r, sb, sr = calculate_node_statistics(G, max_generation)
    
    print(f"Final +1 spins: {final_plus1}")
    print(f"Final -1 spins: {final_minus1}")
    print(f'Final Energy: {final_energy}')
    print(f'Final Total Magnetisation: {final_total_magnetisation}')
    print(f'Final Magnetisation per Spin: {final_magnetisation_per_spin}')
    print(f'Final Blue Node Counts per Generation: {b}')
    print(f'Final Red Node Counts per Generation: {r}')
    print(f'Final Longest Blue Node Index per Generation: {sb}')
    print(f'Final Longest Red Node Index per Generation: {sr}')

if __name__ == "__main__":
    order = 2
    max_generation = 3
    p = 0
    steps = 1000000
    run_simulation(order, max_generation, p, steps)