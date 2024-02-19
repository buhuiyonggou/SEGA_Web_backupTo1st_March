from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import networkx as nx
import os

from pyvis.network import Network
from werkzeug.utils import secure_filename
from graph_utils import load_graph_data, remove_low_degree_nodes, compute_centrality_and_communities, draw_graph_with_pyvis, draw_shortest_path_graph, invert_weights

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
    if request.method == 'POST':
        top_n = int(request.form.get('topN', 10))
    else:
        top_n = 10

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    _, file_extension = os.path.splitext(filename)
    _, G = load_graph_data(filepath, file_extension)

    # remove_low_degree_nodes(G)

    centrality, community_map = compute_centrality_and_communities(G)
    communities = nx.community.louvain_communities(G, weight='weight')

    community_scores = {i: sum(centrality[node] for node in com) for i, com in enumerate(communities)}
    top_communities = sorted(community_scores, key=community_scores.get, reverse=True)[:top_n]

    top_nodes = set().union(*(communities[i] for i in top_communities))
    H = G.subgraph(top_nodes)

    graph_html_path = draw_graph_with_pyvis(H, centrality, community_map)

    return render_template('index.html', graph_html_path=graph_html_path, filename=filename)


@app.route('/find_shortest_path', methods=['POST'])
def find_shortest_path():
    data = request.get_json()
    filename = data['filename']
    nodeStart = data['nodeStart']
    nodeEnd = data['nodeEnd']

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    _, file_extension = os.path.splitext(filename)
    _, G = load_graph_data(filepath, file_extension)

    # 反转权重以反映亲密度
    H = invert_weights(G)

    # 计算最短路径
    path = nx.shortest_path(H, source=nodeStart, target=nodeEnd, weight='weight')

    # 绘制最短路径图
    unique_filename = draw_shortest_path_graph(H, path)

    # 返回生成的图表的路径供前端使用
    return jsonify({'graph_html_path': f'/static/{unique_filename}'})


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
    # app.run(host='0.0.0.0', debug=True)
