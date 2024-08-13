import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt

def add_edges(graph, node, generation, max_generation, g):
    if generation < max_generation:
        num_children = generation + 2
        children = [((generation + 1), node[1] * num_children + i) for i in range(num_children)]
        for child in children:
            if generation + 1 == g:
                graph.add_node(child, spin=-1, fixed=True, boundary=True)  # Set boundary nodes to blue (-1)
            else:
                spin = np.random.choice([-1, 1])
                fixed = False
                graph.add_node(child, spin=spin, fixed=fixed, boundary=False)
            graph.add_edge(node, child)
            add_edges(graph, child, generation + 1, max_generation, g)

def initialize_tree(order, max_generation, g, p):
    G = nx.Graph()
    root = (0, 0)
    spin = np.random.choice([-1, 1])
    fixed = False
    G.add_node(root, spin=spin, fixed=fixed, boundary=False)
    
    add_edges(G, root, 0, max_generation, g)
    
    non_boundary_nodes = [node for node in G.nodes if not G.nodes[node].get('boundary', False)]
    total_non_boundary_nodes = len(non_boundary_nodes)
    
    if p > 0:
        num_fixed_nodes = int(p * total_non_boundary_nodes)
        fixed_nodes = random.sample(non_boundary_nodes, num_fixed_nodes)
        for node in fixed_nodes:
            G.nodes[node]['spin'] = 1
            G.nodes[node]['fixed'] = True
    
    return G

def metropolis_step(G, beta, J=0.65):
    nodes = list(G.nodes())
    random.shuffle(nodes)
    for node in nodes:
        if G.nodes[node].get('fixed', False) or G.nodes[node].get('boundary', False):
            continue
        S = G.nodes[node]['spin']
        neighbors = list(G.neighbors(node))
        neighbor_spins = sum(G.nodes[n]['spin'] for n in neighbors)
        dE = 2 * J * S * neighbor_spins
        if dE < 0 or random.random() < np.exp(-dE * beta):
            G.nodes[node]['spin'] = -S

def simulate(G, T, steps, J=0.65):
    beta = 1.0 / T
    plot_tree(G, title='Initial State')  # Plot initial state
    for _ in range(steps):
        metropolis_step(G, beta, J)
    plot_tree(G, title='Final State')  # Plot final state

def plot_tree(G, title='Tree'):
    pos = nx.spring_layout(G, seed=42)  # Use spring layout with a fixed seed for better visualization
    spins = nx.get_node_attributes(G, 'spin')
    boundary = nx.get_node_attributes(G, 'boundary')
    
    colors = []
    for node in G.nodes:
        if boundary.get(node, False):
            colors.append('green')  # Boundary nodes are green
        else:
            colors.append('red' if spins[node] == 1 else 'blue')
    
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, node_color=colors, with_labels=True, node_size=500, font_color='white')
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    order = 2
    max_generation = 5
    p = 0.2
    g = max_generation  # Use max_generation to ensure correct layer depth
    G = initialize_tree(order, max_generation, g, p)
    T = 0.001  # Temperature
    steps = 1000  # Number of Metropolis steps
    simulate(G, T, steps)
