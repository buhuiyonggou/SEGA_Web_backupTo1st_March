import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
from pyvis.network import Network
from fractions import Fraction


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
        net.add_node(node, title=str(node), size=centrality[node] * 500, group=community_map[node], color=color)

    # 计算图中所有边的最大权重
    max_weight = max(data['weight'] for _, _, data in X.edges(data=True))

    # 只添加前 100 个节点之间的边，并根据权重相对于最大权重的比例设置边的宽度
    for source, target, data in X.edges(data=True):
        if source in top_nodes and target in top_nodes:
            weight = data.get('weight', 1)  # 默认权重为 1，如果没有指定权重
            edge_width = weight / max_weight * 10  # 根据权重归一化计算宽度
            net.add_edge(source, target, width=edge_width)

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


def draw_shortest_path_graph(G, path):
    net = Network(height="300px", width="100%", bgcolor="#ffffff", font_color="black")

    # 添加路径中的节点和边
    previous_node = None
    for node in path:
        net.add_node(node, title=str(node), label=str(node))
        if previous_node is not None:
            weight = G[previous_node][node].get('weight', 1)  # 获取边的权重，默认为1
            # 转换权重为分数表示
            edge_weight = str(Fraction(weight).limit_denominator())
            net.add_edge(previous_node, node, title=str(edge_weight), label=str(edge_weight))
        previous_node = node

    # 优化布局参数
    net.options.physics.barnesHut.springLength = 200
    net.options.physics.barnesHut.springConstant = 0.05
    net.options.physics.barnesHut.damping = 0.09
    net.options.physics.barnesHut.centralGravity = 0.3
    net.options.physics.barnesHut.gravitationalConstant = -800
    net.options.physics.maxVelocity = 50
    net.options.physics.minVelocity = 0.1

    # net.show_buttons(filter_=['manipulation', 'physics'])  # 这行被移除，不再显示操作按钮

    unique_filename = f"shortest_path.html"
    filepath = os.path.join('static', unique_filename)
    net.write_html(filepath)

    return unique_filename


def invert_weights(G):
    H = nx.Graph()
    for u, v, data in G.edges(data=True):
        # 确保权重为正数，避免除零错误
        if data['weight'] > 0:
            H.add_edge(u, v, weight=1.0 / data['weight'])
        else:
            # 对于权重为0或未定义的情况，可以设置一个大的权重值
            H.add_edge(u, v, weight=float('inf'))
    return H

#
# def draw_graph(G, centrality, community_map, fig_size=(5, 3), node_scale=3000, title=""):
#     fig, ax = plt.subplots(figsize=fig_size)
#     pos = nx.spring_layout(G, k=0.5, seed=4572321)
#     node_color = [community_map.get(n, 0) for n in G.nodes()]
#     node_size = [centrality.get(n, 0) * node_scale for n in G.nodes()]
#     nx.draw_networkx(
#         G,
#         pos=pos,
#         with_labels=True,
#         node_color=node_color,
#         node_size=node_size,
#         edge_color="gainsboro",
#         alpha=0.4,
#         ax=ax,
#         font_size=3
#     )
#     set_graph_title_and_legend(ax, fig, title=title)
#     return fig, ax
#
#
# def set_graph_title_and_legend(ax, fig, title=""):
#     font_title = {"color": "black", "fontweight": "bold", "fontsize": 5}
#     font_legend = {"color": "red", "fontweight": "bold", "fontsize": 3}
#     ax.set_title(title, fontdict=font_title)
#     ax.text(0.80, 0.10, "Node color = Community structure", horizontalalignment="center", transform=ax.transAxes,
#             fontdict=font_legend)
#     ax.text(0.80, 0.06, "Node size = PageRank centrality", horizontalalignment="center", transform=ax.transAxes,
#             fontdict=font_legend)
#     ax.margins(0.1, 0.05)
#     fig.tight_layout()
#     plt.axis("off")
#
#
# def save_fig_to_base64(fig):
#     img = BytesIO()
#     fig.savefig(img, format='png', dpi=300)
#     plt.close(fig)
#     img.seek(0)
#     return base64.b64encode(img.getvalue()).decode('utf8')
#
# def compute_centrality_and_communities(X):
#     centrality = nx.pagerank(X, weight='weight')
#     communities = nx.community.louvain_communities(X, weight='weight')
#     community_map = {node: i for i, community in enumerate(communities) for node in community}
#     return centrality, community_map
#
#
# def compute_betweenness_and_louvain(X):
#     # 计算介数中心性
#     centrality = nx.betweenness_centrality(X, weight='weight')
#     # 使用Louvain方法计算社区结构
#     communities = nx.community.louvain_communities(X, weight='weight')
#     # 构建社区映射字典：节点 -> 社区编号
#     community_map = {node: i for i, community in enumerate(communities) for node in community}
#     return centrality, community_map

