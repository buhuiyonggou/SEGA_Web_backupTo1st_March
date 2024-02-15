from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
from werkzeug.utils import secure_filename
from pyvis.network import Network

app = Flask(__name__)

# 配置上传文件夹和允许上传的文件类型
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'BabaYaga'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 检查是否有文件在请求的文件部分
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # 如果用户没有选择文件，浏览器会提交一个没有文件名的空部分
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return render_template('analyze.html', filename=filename)
    return render_template('upload.html')


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


def compute_centrality_and_communities(G):
    centrality = nx.pagerank(G, weight='weight')
    communities = nx.community.louvain_communities(G, weight='weight')
    community_map = {node: i for i, community in enumerate(communities) for node in community}
    return centrality, community_map


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


def draw_graph_with_pyvis(G, centrality, community_map):
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black")

    # 根据 centrality 值对节点进行排序，选择前 100 个
    top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:100]

    # 准备社区颜色
    community_colors = ["#FFA07A", "#20B2AA", "#778899", "#9370DB", "#FFD700", "#FF6347", "#3CB371", "#F08080",
                        "#00FA9A", "#BDB76B", "#FF00FF"]

    # 添加前 100 个节点到 Pyvis 网络，并设置颜色
    for node in top_nodes:
        color = community_colors[community_map[node] % len(community_colors)]
        net.add_node(node, title=str(node), size=centrality[node] * 1000, group=community_map[node], color=color)

    # 只添加前 500 个节点之间的边
    for source, target in G.edges(top_nodes):
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

    # 指定生成的 HTML 文件路径
    filepath = os.path.join('static', 'graph.html')
    net.write_html(filepath)

    return 'graph.html'


@app.route('/show_graph/<filename>')
def network_graph(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    _, file_extension = os.path.splitext(filename)

    _, G = load_graph_data(filepath, file_extension)

    remove_low_degree_nodes(G)

    centrality, community_map = compute_centrality_and_communities(G)

    # 使用 Pyvis 画图
    graph_html_path = draw_graph_with_pyvis(G, centrality, community_map)

    # 将 HTML 文件路径传递给模板，而不是图像的 base64 编码
    return render_template('index.html', graph_html_path=graph_html_path, filename=filename)


@app.route('/show_top_communities/<filename>', methods=['GET', 'POST'])
def show_top_communities(filename):
    if request.method == 'POST':
        top_n = int(request.form.get('topN', 10))  # 默认为Top 10，如果用户没有输入值
    else:
        top_n = 10  # 如果是GET请求，也使用默认值Top 10
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    _, file_extension = os.path.splitext(filename)

    _, G = load_graph_data(filepath, file_extension)
    remove_low_degree_nodes(G)
    centrality, community_map = compute_centrality_and_communities(G)

    communities = nx.community.louvain_communities(G, weight='weight')
    # 根据用户输入选择Top N个社区
    community_scores = {i: sum(centrality[node] for node in com) for i, com in enumerate(communities)}
    top_communities = sorted(community_scores, key=community_scores.get, reverse=True)[:top_n]

    # 为前10个社区创建子图
    top_nodes = set().union(*(communities[i] for i in top_communities))
    H = G.subgraph(top_nodes)

    fig, ax = draw_graph(H, centrality, community_map, title="Top N Communities Analysis")
    plot_url = save_fig_to_base64(fig)

    return render_template('index.html', plot_url=plot_url, filename=filename)


@app.route('/delete_node/<filename>', methods=['POST'])
def delete_node(filename):
    # 记录接收到的请求和文件名
    node_id = request.form['nodeId']  # 获取用户输入的节点ID

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    _, file_extension = os.path.splitext(filename)

    _, G = load_graph_data(filepath, file_extension)

    # 删除节点及其相连的边
    if node_id in G:
        G.remove_node(node_id)
        flash(f"Node {node_id} deleted.")
    else:
        flash(f"Node {node_id} not found.")
        return redirect(url_for('network_graph', filename=filename))

    remove_low_degree_nodes(G)
    centrality, community_map = compute_centrality_and_communities(G)

    fig, ax = draw_graph(G, centrality, community_map, title="PageRank and Louvain")
    plot_url = save_fig_to_base64(fig)

    return render_template('index.html', plot_url=plot_url, filename=filename)


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
    # app.run(host='0.0.0.0', debug=True)
