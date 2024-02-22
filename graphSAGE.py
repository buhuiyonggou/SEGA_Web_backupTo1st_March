import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import MinMaxScaler

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        # self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        # x = self.conv2(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# Initialize the GraphSAGE model
model = GraphSAGE(in_channels=feature_index.shape[1], hidden_channels=16, out_channels=8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

data = Data(feature_index=feature_index, edge_index=edge_index)

# Training loop
for epoch in range(150):  # Number of epochs
    model.train()
    optimizer.zero_grad()
    out = model(data.feature_index, data.edge_index)
    # Example loss calculation; adjust according to your specific task
    loss = ((out[data.edge_index[0]] - out[data.edge_index[1]]) ** 2).mean()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch + 10}, Loss: {loss.item()}')

with torch.no_grad():
    embeddings = model(feature_index, edge_index)
    new_weights = torch.norm(embeddings[edge_index[0]] - embeddings[edge_index[1]], dim=1)

# Initialize the scaler
scaler = MinMaxScaler()

# Reshape new_weights for scaling - sklearn's MinMaxScaler expects a 2D array
weights_reshaped = new_weights.numpy().reshape(-1, 1)

# Apply the scaler to the weights
scaled_weights = scaler.fit_transform(weights_reshaped).flatten()

# Create a DataFrame to export
edges_with_weights = pd.DataFrame(edge_index.t().numpy(), columns=['source', 'target'])

# Update the DataFrame with scaled weights
edges_with_weights['weight'] = scaled_weights

# Use id to map names
edges_with_weights['source'] = edges_with_weights['source'].map(index_to_name_mapping)
edges_with_weights['target'] = edges_with_weights['target'].map(index_to_name_mapping)

# Save the DataFrame to a CSV file
output_path = '/content/edges_with_weights.csv'
edges_with_weights.to_csv(output_path, index=False)

print(f'Edge weights saved to {output_path}')