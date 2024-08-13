import networkx as nx
import matplotlib.pyplot as plt

def add_edges(graph, node, generation, max_generation):
    if generation < max_generation:
        num_children = generation + 2
        children = [((generation + 1), (node[1] - 1) * num_children + i + 1) for i in range(num_children)]
        for child in children:
            graph.add_edge(node, child)
            add_edges(graph, child, generation + 1, max_generation)

def hierarchy_pos(G, root, width=1, vert_gap=0.2, vert_loc=1, xcenter=0.5):
    pos = {}
    def _hierarchy_pos(G, node, xcenter, width, vert_loc, pos, parent=None):
        pos[node] = (xcenter, vert_loc)
        children = list(G.neighbors(node))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, nextx, dx, vert_loc - vert_gap, pos, node)
        return pos
    return _hierarchy_pos(G, root, xcenter, width, vert_loc, pos)

def plot_cayley_tree(max_generation):
    G = nx.Graph()
    root = (0, 0)
    G.add_node(root)
    add_edges(G, root, 0, max_generation)
    
    pos = hierarchy_pos(G, root)
    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, labels={node: f"{node}" for node in G.nodes()}, 
            node_color='lightblue', node_size=500, font_size=10, font_weight='bold', ax=ax)
    plt.title(f'Cayley Tree with Increasing Children per Generation')
    plt.show()

# max_generation represents the level of tree
plot_cayley_tree(max_generation=3)
