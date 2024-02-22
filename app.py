import os
import pandas as pd
import networkx as nx
from flask import Flask, render_template, request, redirect, flash, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from algorithms import calculate_centrality, detect_communities
from graph_utils import draw_graph_with_pyvis, draw_shortest_path_graph, invert_weights
from pyecharts import options as opts
from pyecharts.charts import Tree
from pyecharts.globals import ThemeType
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
import json

app = Flask(__name__)

# Configuration for the file upload folder and allowed file types
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'BabaYaga'


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Convert adjacency matrix to edge list
def adjacency_to_edgelist(adj_matrix_df):
    edges = []
    for i, row in adj_matrix_df.iterrows():
        for j, weight in row.items():  # Changed from iteritems() to items()
            if weight != 0 and i != j:  # Assuming no self-loops and non-zero weight
                edges.append((i, j, weight))
    return pd.DataFrame(edges, columns=['Source', 'Target', 'Weight'])


# Load graph data from a file
def load_graph_data(filepath, file_extension):
    try:
        if file_extension.lower() == '.csv':
            df = pd.read_csv(filepath)
        elif file_extension.lower() == '.xlsx':
            adj_matrix_df = pd.read_excel(filepath, index_col=0)
            df = adjacency_to_edgelist(adj_matrix_df)
        else:
            raise ValueError("Unsupported file type.")

        # Check if required columns are present
        if not {'Source', 'Target', 'Weight'}.issubset(df.columns):
            raise ValueError("Dataframe must contain 'Source', 'Target', and 'Weight' columns.")

        # Sort by 'Weight' in descending order and select top 5000 rows
        # df = df.sort_values(by='Weight', ascending=False).head(3000)

        # Create graph from dataframe
        G = nx.Graph()
        for _, row in df.iterrows():
            G.add_edge(row['Source'], row['Target'], weight=row['Weight'])

        return df, G
    except Exception as e:
        # Handle any errors that occur during data loading
        flash(str(e))  # Display the error message to the user
        return None, None  # Return None values to indicate failure

@app.route('/', methods=['GET', 'POST'])
def upload_user_data():
    """Handle file uploads."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

    return render_template('graphSAGE.html')

@app.route('/analyze')
def analyze():
    return render_template('analyze.html')

@app.route('/upload_to_vis', methods=['GET', 'POST'])
def upload_file():
    """Handle file uploads."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
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
    """Display the network graph based on selected algorithms."""
    centrality_algo = request.args.get('centrality', 'pagerank')
    community_algo = request.args.get('community', 'louvain')

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    _, file_extension = os.path.splitext(filename)

    _, G = load_graph_data(filepath, file_extension)

    if G is None:  # Check if G is None, indicating an error occurred
        # flash('File format error. Please upload a valid file.', 'danger')
        return redirect(url_for('upload_file'))  # Redirect the user to upload page

    # remove_low_degree_nodes(G)

    centrality = calculate_centrality(G, centrality_algo)
    communities = detect_communities(G, community_algo)
    community_map = {node: i for i, community in enumerate(communities) for node in community}

    graph_html_path = draw_graph_with_pyvis(G, centrality, community_map)

    return render_template('index.html', graph_html_path=graph_html_path, filename=filename,
                           community_algo=community_algo, centrality_algo=centrality_algo)


@app.route('/show_top_communities/<filename>', methods=['GET', 'POST'])
def show_top_communities(filename):
    """Show top communities based on the selected algorithms."""
    if request.method == 'POST':
        top_n = int(request.form.get('topN', 10))
        centrality_algo = request.form.get('centrality', 'pagerank')  # Use request.form for POST data
        community_algo = request.form.get('community', 'louvain')  # Use request.form for POST data
    else:
        top_n = 10
        centrality_algo = 'pagerank'
        community_algo = 'louvain'

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    _, file_extension = os.path.splitext(filename)
    _, G = load_graph_data(filepath, file_extension)

    if G is None:  # Check if G is None, indicating an error occurred
        # flash('File format error. Please upload a valid file.', 'danger')
        return redirect(url_for('upload_file'))  # Redirect the user to upload page

    # remove_low_degree_nodes(G)

    centrality = calculate_centrality(G, centrality_algo)
    communities = detect_communities(G, community_algo)
    community_map = {node: i for i, community in enumerate(communities) for node in community}

    community_scores = {i: sum(centrality[node] for node in com) for i, com in enumerate(communities)}
    top_communities = sorted(community_scores, key=community_scores.get, reverse=True)[:top_n]

    top_nodes = set().union(*(communities[i] for i in top_communities))
    H = G.subgraph(top_nodes)

    graph_html_path = draw_graph_with_pyvis(H, centrality, community_map)

    return render_template('index.html', graph_html_path=graph_html_path, filename=filename,
                           community_algo=community_algo, centrality_algo=centrality_algo)


@app.route('/find_shortest_path', methods=['POST'])
def find_shortest_path():
    """Find and display the shortest path between two nodes."""
    data = request.get_json()
    filename = data['filename']
    nodeStart = data['nodeStart']
    nodeEnd = data['nodeEnd']

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    _, file_extension = os.path.splitext(filename)
    _, G = load_graph_data(filepath, file_extension)

    # Check if the specified nodes exist in the graph
    if nodeStart not in G or nodeEnd not in G:
        return jsonify({'error': 'One or both of the specified nodes do not exist in the graph.'})

    # Invert weights to reflect closeness instead of distance
    H = invert_weights(G)

    path = nx.shortest_path(H, source=nodeStart, target=nodeEnd, weight='weight')
    unique_filename = draw_shortest_path_graph(H, path)

    return jsonify({'graph_html_path': f'/static/{unique_filename}'})


@app.route('/show_dendrogram/<filename>')
def show_dendrogram(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    _, file_extension = os.path.splitext(filename)

    # 加载数据并创建图G
    df, G = load_graph_data(file_path, file_extension)

    if G is None:
        flash('Error loading graph data.', 'danger')
        return redirect(url_for('upload_file'))

    # 使用Graph的邻接矩阵，但需要转换为距离矩阵
    # 这里我们直接从G计算距离矩阵
    distance_matrix = nx.floyd_warshall_numpy(G, weight='weight')
    # 因为floyd_warshall_numpy返回的是numpy数组，我们需要将其转换为适合linkage函数的格式
    Z = linkage(squareform(distance_matrix, checks=False), method='complete')

    # 转换Z为树结构的JSON
    dendrogram_json = convert_to_dendrogram_json(Z, list(G.nodes()))

    # 保存dendrogram_json到文件
    dendrogram_json_path = os.path.join('static', 'dendrogram.json')
    with open(dendrogram_json_path, 'w') as f:
        json.dump(dendrogram_json, f)

    # 使用Pyecharts生成树状图
    tree_chart = (
        Tree(init_opts=opts.InitOpts(width="1200px", height="900px", theme=ThemeType.LIGHT))
        .add("", [dendrogram_json], collapse_interval=10, initial_tree_depth=30, is_roam=True,
             symbol="circle", label_opts=opts.LabelOpts(font_size=7)
             )
        # .set_global_opts(title_opts=opts.TitleOpts(title="Dendrogram"))
    )

    # 保存树状图为HTML文件
    dendrogram_html_filename = 'dendrogram_chart.html'
    tree_chart.render(path=os.path.join('static', dendrogram_html_filename))

    # 重定向到树状图页面
    return render_template('dendrogram.html', dendrogram_html_filename=dendrogram_html_filename, filename=filename)


def convert_to_dendrogram_json(Z, labels):
    # Convert the linkage matrix into a tree structure.
    tree = to_tree(Z, rd=False)

    def count_leaves(node):
        # Recursively count the leaves under a node
        if node.is_leaf():
            return 1
        return count_leaves(node.left) + count_leaves(node.right)

    # Recursive function to build the JSON structure
    def build_json(node):
        if node.is_leaf():
            # For leaf nodes, use the provided labels
            return {"name": labels[node.id]}
        else:
            # For internal nodes, generate a name that includes the cluster size
            size = count_leaves(node)
            name = f"Cluster of {size}"
            # Recursively build the JSON for children
            return {
                "name": name,
                "children": [build_json(node.left), build_json(node.right)]
            }

    # Build and return the JSON structure
    return build_json(tree)


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
    # app.run(host='0.0.0.0', debug=True)

# @app.route('/delete_node/<filename>', methods=['POST'])
# def delete_node(filename):
#     node_id = request.form['nodeId']  # 获取用户输入的节点ID
#
#     filepath = os.path.join(UPLOAD_FOLDER, filename)
#     _, file_extension = os.path.splitext(filename)
#
#     _, G = load_graph_data(filepath, file_extension)
#
#     if node_id in G:
#         G.remove_node(node_id)
#         flash(f"Node {node_id} deleted.")
#     else:
#         flash(f"Node {node_id} not found.")
#         return redirect(url_for('network_graph', filename=filename))
#
#     # remove_low_degree_nodes(G)
#     centrality, community_map = compute_centrality_and_communities(G)
#
#     # fig, ax = draw_graph(G, centrality, community_map, title="PageRank and Louvain")
#     # plot_url = save_fig_to_base64(fig)
#     #
#     # return render_template('index.html', plot_url=plot_url, filename=filename)
#     graph_html_path = draw_graph_with_pyvis(G, centrality, community_map)
#
#     return render_template('index.html', graph_html_path=graph_html_path, filename=filename)
