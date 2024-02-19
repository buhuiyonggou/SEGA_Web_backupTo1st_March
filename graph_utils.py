import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
from pyvis.network import Network


def adjacency_to_edgelist(adj_matrix_df):
    edges = []
    for i, row in adj_matrix_df.iterrows():
        for j, weight in row.items():  # Changed from iteritems() to items()
            if weight != 0 and i != j:  # Assuming no self-loops and non-zero weight
                edges.append((i, j, weight))
    return pd.DataFrame(edges, columns=['Source', 'Target', 'Weight'])


def load_graph_data(filepath, file_extension):
    if file_extension.lower() == '.csv':
        df = pd.read_csv(filepath)
    elif file_extension.lower() == '.xlsx':
        adj_matrix_df = pd.read_excel(filepath, index_col=0)
        df = adjacency_to_edgelist(adj_matrix_df)
    else:
        raise ValueError("Unsupported file type.")

    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['Weight'])

    return df, G


def remove_low_degree_nodes(G, min_degree=5):
    low_degree_nodes = [node for node, degree in G.degree() if degree < min_degree]
    G.remove_nodes_from(low_degree_nodes)


def compute_centrality_and_communities(X):
    centrality = nx.pagerank(X, weight='weight')
    communities = nx.community.louvain_communities(X, weight='weight')
    community_map = {node: i for i, community in enumerate(communities) for node in community}
    return centrality, community_map


def draw_graph_with_pyvis(X, centrality, community_map):
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black")

    # 根据 centrality 值对 H 中的节点进行排序，选择前 100 个
    top_nodes = sorted((node for node in X.nodes()), key=lambda node: centrality.get(node, 0), reverse=True)[:100]

    # 准备社区颜色
    community_colors = ["#FFA07A", "#20B2AA", "#778899", "#9370DB", "#FFD700", "#FF6347", "#3CB371", "#F08080",
                        "#00FA9A", "#BDB76B", "#FF00FF"]

    # 添加前 100 个节点到 Pyvis 网络，并设置颜色
    for node in top_nodes:
        color = community_colors[community_map[node] % len(community_colors)]
        net.add_node(node, title=str(node), size=centrality[node] * 1000, group=community_map[node], color=color)

    # 只添加前 500 个节点之间的边
    for source, target in X.edges(top_nodes):
        if source in top_nodes and target in top_nodes:
            net.add_edge(source, target)

    # 优化布局参数
    net.options.physics.barnesHut.springLength = 2000
    net.options.physics.barnesHut.springConstant = 0.02
    net.options.physics.barnesHut.damping = 0.09
    net.options.physics.barnesHut.centralGravity = 0.2
    net.options.physics.barnesHut.gravitationalConstant = -1000
    net.options.physics.maxVelocity = 5
    net.options.physics.minVelocity = 0.1

    # Enable node deletion, node add, edge add, etc
    net.show_buttons(filter_=['manipulation', 'physics'])

    # 指定生成的 HTML 文件路径
    filepath = os.path.join('static', 'graph.html')
    net.write_html(filepath)

    return 'graph.html'


def draw_graph(G, centrality, community_map, fig_size=(5, 3), node_scale=3000, title=""):
    fig, ax = plt.subplots(figsize=fig_size)
    pos = nx.spring_layout(G, k=0.5, seed=4572321)
    node_color = [community_map.get(n, 0) for n in G.nodes()]
    node_size = [centrality.get(n, 0) * node_scale for n in G.nodes()]
    nx.draw_networkx(
        G,
        pos=pos,
        with_labels=True,
        node_color=node_color,
        node_size=node_size,
        edge_color="gainsboro",
        alpha=0.4,
        ax=ax,
        font_size=3
    )
    set_graph_title_and_legend(ax, fig, title=title)
    return fig, ax


def set_graph_title_and_legend(ax, fig, title=""):
    font_title = {"color": "black", "fontweight": "bold", "fontsize": 5}
    font_legend = {"color": "red", "fontweight": "bold", "fontsize": 3}
    ax.set_title(title, fontdict=font_title)
    ax.text(0.80, 0.10, "Node color = Community structure", horizontalalignment="center", transform=ax.transAxes,
            fontdict=font_legend)
    ax.text(0.80, 0.06, "Node size = PageRank centrality", horizontalalignment="center", transform=ax.transAxes,
            fontdict=font_legend)
    ax.margins(0.1, 0.05)
    fig.tight_layout()
    plt.axis("off")


def save_fig_to_base64(fig):
    img = BytesIO()
    fig.savefig(img, format='png', dpi=300)
    plt.close(fig)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')

