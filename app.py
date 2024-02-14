from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 配置上传文件夹和允许上传的文件类型
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


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
            return redirect(url_for('network_graph', filename=filename))
    return render_template('upload.html')


def adjacency_to_edgelist(adj_matrix_df):
    edges = []
    for i, row in adj_matrix_df.iterrows():
        for j, weight in row.items():  # Changed from iteritems() to items()
            if weight != 0 and i != j:  # Assuming no self-loops and non-zero weight
                edges.append((i, j, weight))
    return pd.DataFrame(edges, columns=['Source', 'Target', 'Weight'])



@app.route('/show_graph/<filename>')
def network_graph(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    _, file_extension = os.path.splitext(filename)

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

    # 移除低度节点
    low_degree_nodes = [node for node, degree in dict(G.degree()).items() if degree < 5]
    G.remove_nodes_from(low_degree_nodes)

    # 计算PageRank中心性
    centrality = nx.pagerank(G, weight='weight')

    # 使用NetworkX的内置Louvain方法计算社区结构
    community_map = {}
    communities = nx.community.louvain_communities(G, weight='weight')
    for i, community in enumerate(communities):
        for node in community:
            community_map[node] = i

    # 绘图
    fig, ax = plt.subplots(figsize=(5, 3))
    pos = nx.spring_layout(G, k=0.15, seed=4572321)
    node_color = [community_map[n] for n in G.nodes()]
    node_size = [centrality[n] * 3000 for n in G.nodes()]
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
    # 设置标题和图例说明
    font_title = {"color": "black", "fontweight": "bold", "fontsize": 5}
    font_legend = {"color": "red", "fontweight": "bold", "fontsize": 3}
    ax.set_title("PageRank and Louvain", fontdict=font_title)

    # 添加图例说明
    ax.text(0.80, 0.10, "Node color = Community structure", horizontalalignment="center", transform=ax.transAxes,
            fontdict=font_legend)
    ax.text(0.80, 0.06, "Node size = PageRank centrality", horizontalalignment="center", transform=ax.transAxes,
            fontdict=font_legend)

    # 优化边界和标签可读性
    ax.margins(0.1, 0.05)
    fig.tight_layout()
    plt.axis("off")

    # 将绘制的图转换为Base64编码的图片
    img = BytesIO()
    plt.savefig(img, format='png', dpi=300)
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('index.html', plot_url=plot_url)


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
