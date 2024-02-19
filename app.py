from flask import Flask, render_template, request, redirect, url_for, flash
import networkx as nx
import os
from werkzeug.utils import secure_filename
from graph_utils import load_graph_data, remove_low_degree_nodes, compute_centrality_and_communities, draw_graph, draw_graph_with_pyvis, save_fig_to_base64

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

@app.route('/show_graph/<filename>')
def network_graph(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    _, file_extension = os.path.splitext(filename)

    _, G = load_graph_data(filepath, file_extension)

    # remove_low_degree_nodes(G)

    centrality, community_map = compute_centrality_and_communities(G)

    # 使用 Pyvis 画图
    graph_html_path = draw_graph_with_pyvis(G, centrality, community_map)

    # 将 HTML 文件路径传递给模板，而不是图像的 base64 编码
    return render_template('index.html', graph_html_path=graph_html_path, filename=filename)


@app.route('/show_top_communities/<filename>', methods=['GET', 'POST'])
def show_top_communities(filename):
    print("Starting top communities analysis for", filename)

    if request.method == 'POST':
        top_n = int(request.form.get('topN', 10))
    else:
        top_n = 10
    print("Selected top N communities:", top_n)

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    _, file_extension = os.path.splitext(filename)
    _, G = load_graph_data(filepath, file_extension)

    # remove_low_degree_nodes(G)

    centrality, community_map = compute_centrality_and_communities(G)
    communities = nx.community.louvain_communities(G, weight='weight')
    print("Total communities found:", len(communities))

    community_scores = {i: sum(centrality[node] for node in com) for i, com in enumerate(communities)}
    top_communities = sorted(community_scores, key=community_scores.get, reverse=True)[:top_n]

    top_nodes = set().union(*(communities[i] for i in top_communities))
    H = G.subgraph(top_nodes)
    print("Nodes in the subgraph (H):", len(H.nodes()))

    graph_html_path = draw_graph_with_pyvis(H, centrality, community_map)

    return render_template('index.html', graph_html_path=graph_html_path, filename=filename)


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

    # remove_low_degree_nodes(G)
    centrality, community_map = compute_centrality_and_communities(G)

    # fig, ax = draw_graph(G, centrality, community_map, title="PageRank and Louvain")
    # plot_url = save_fig_to_base64(fig)
    #
    # return render_template('index.html', plot_url=plot_url, filename=filename)
    # 使用 Pyvis 画图
    graph_html_path = draw_graph_with_pyvis(G, centrality, community_map)

    # 将 HTML 文件路径传递给模板，而不是图像的 base64 编码
    return render_template('index.html', graph_html_path=graph_html_path, filename=filename)


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
    # app.run(host='0.0.0.0', debug=True)
