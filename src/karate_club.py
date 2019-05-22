import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random

np.random.seed(555)
random.seed(555)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def draw_original_graph():
    original_g = nx.karate_club_graph()

    plt.figure(figsize=(6, 4.5))
    node_color = ['brown' if not j else 'skyblue' for j in labels]

    pos = nx.spring_layout(original_g)
    nx.draw_networkx_nodes(original_g, pos=pos, with_labels=False, node_color=node_color, node_size=50)
    nx.draw_networkx_edges(original_g, pos=pos, edgelist=added_edges, style='dotted', edge_color='silver')
    nx.draw_networkx_edges(original_g, pos=pos, edgelist=original_g.edges, edge_color='grey')

    plt.axis('off')
    plt.savefig('karate.pdf', bbox_inches='tight')
    exit(0)


def plot(index, pos):
    plt.subplot(141 + index).set_title('%d-layer' % (index + 1))
    plt.xticks([])
    plt.yticks([])
    for e in G.edges:
        plt.plot([pos[e[0], 0], pos[e[1], 0]], [pos[e[0], 1], pos[e[1], 1]], color='silver', lw=0.5, zorder=1)
    plt.scatter([pos[~labels, 0]], [pos[~labels, 1]], color='brown', s=80, zorder=2)
    plt.scatter([pos[labels, 0]], [pos[labels, 1]], color='skyblue', s=80, zorder=2)


if __name__ == '__main__':
    add_noise = False
    gcn_lpa = False

    G = nx.karate_club_graph()
    n_nodes = G.number_of_nodes()

    features = np.eye(n_nodes)
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,
                       0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).astype(np.bool)

    added_edges = []
    if add_noise:
        i = 0
        while True:
            a = random.randint(0, n_nodes - 1)
            b = random.randint(0, n_nodes - 1)
            if labels[a] != labels[b] and not G.has_edge(a, b):
                G.add_edge(a, b)
                added_edges.append([a, b])
                i += 1
            if i == 20:
                break

    adj = nx.adjacency_matrix(G).todense().astype(np.float)
    adj += np.eye(n_nodes)

    if gcn_lpa:
        for i in range(n_nodes):
            for j in range(i, n_nodes):
                if G.has_edge(i, j) and labels[i] != labels[j]:
                    adj[i, j] /= 10
    adj /= np.sum(adj, axis=-1)

    # if you want to visualize the karate club graph, uncomment the next line
    # draw_original_graph()

    plt.figure(figsize=(15, 2.5))
    # plt.figure(figsize=(6, 4.5))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    h = features
    for i in range(4):
        w = np.random.rand(n_nodes, 2) * 2 - 1 if i == 0 else np.random.rand(2, 2) * 2 - 1
        z = np.matmul(adj, h)
        h = sigmoid(np.matmul(z, w))
        plot(i, h)

    if add_noise:
        if gcn_lpa:
            plt.savefig('gcn_lpa_noise.pdf', bbox_inches='tight')
        else:
            plt.savefig('gcn_noise.pdf', bbox_inches='tight')
    else:
        if gcn_lpa:
            plt.savefig('gcn_lpa.pdf', bbox_inches='tight')
        else:
            plt.savefig('gcn.pdf', bbox_inches='tight')
