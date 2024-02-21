import os
import pandas as pd
import networkx as nx
from flask import Flask, render_template, request, redirect, flash, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from algorithms import calculate_centrality, detect_communities
from graph_utils import draw_graph_with_pyvis, draw_shortest_path_graph, invert_weights

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
        df = df.sort_values(by='Weight', ascending=False).head(3000)

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

    return render_template('index.html', graph_html_path=graph_html_path, filename=filename, community_algo=community_algo, centrality_algo=centrality_algo)


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

    return render_template('index.html', graph_html_path=graph_html_path, filename=filename, community_algo=community_algo, centrality_algo=centrality_algo)


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
