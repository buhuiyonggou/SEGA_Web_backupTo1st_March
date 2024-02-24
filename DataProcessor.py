import pandas as pd
import numpy as np
from itertools import combinations
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from graphSAGE import GraphSAGE

class GraphSAGEProcessor:
    def __init__(self, nodes_folder, edges_folder = None):
        self.nodes_folder = nodes_folder
        self.edges_folder = edges_folder
        
        self.CONNECT_AMONG_SUB_DEPART = 0.25
        self.CONNECT_AMONG_DEPARTMENT = 0.05
        self.CONNECT_AMONG_ORGANZATION = 0.01
        self.COLUMN_ALIGNMENT = {
            'id':['id','emp_id', 'employeeid'],
            'name': ['name', 'full_name', "employee_name", 'first_name'],
            'department': ['department', 'depart', 'sector'],
            'sub-depart':['sub-depart', 'sub-department', 'sub-sector','second-department'],
            'manager': ['manager', 'supervisor', 'manager_name','managername']
        }
        self.NODE_FEATURES = {
            'title': ['title', 'job_title', 'position','job_level'],
            'nationality': ['nationality', 'citizen', 'citizenship'],
            'region': ['region', 'district', 'branch', 'area'],
            'education':['education', 'degree', 'graduate'],
            'salary': ['salary', 'sal'],
            'marriage_status': ['marriage_status', 'marital_status', 'marrage_status'],
            'race': ['race', 'ethnicity'],
            'gender': ['gender', 'sex'],
            'employment_status': ['employment_status', 'job_status', 'employment']
        }

    def load_data(self, file_path):
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='utf-8')  # Handles BOM if present
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format")

        df.columns = df.columns.str.lower()
        return df

    def fetch_data_from_user(self, file_path):
        if file_path is None:
            return
        hr_data = self.load_data(file_path)
        hr_data.fillna("missing", inplace=True)
        
        return hr_data

    # Function to rename columns based on expected variations
    def rename_columns_to_standard(self, df, column_alignment):
        renamed_columns = df.columns.tolist()  # Start with the original column names
        for standard_name, variations in column_alignment.items():
            for variation in variations:
                # Find which variation is in the DataFrame columns (case-insensitive match)
                found_columns = [col for col in df.columns if col.lower() == variation.lower()]
                if found_columns:
                    # Assume only one match is found; rename it to the standard name
                    index_of_found = renamed_columns.index(found_columns[0])
                    renamed_columns[index_of_found] = standard_name
                    break  # Stop looking for other variations if one is found
        return renamed_columns

    def create_index_id_name_mapping(self, hr_data):
        index_to_id_name_mapping = [{
            'index': i,
            'id': row['id'],
            'name': row['name'].title()
        } for i, row in hr_data.iterrows()]

        mapping_df = pd.DataFrame(index_to_id_name_mapping)
        return mapping_df

    # Function to find the actual column name in the DataFrame based on expected variations
    def find_actual_column_name(self, df, possible_names):
        for actual_name in df.columns.str.lower():
            if actual_name in possible_names:
                return actual_name
        return None

    # Mapping from your standardized property names to actual column names in hr_data
    def generate_attributes(self, attrs, hr_data): 
        property_to_actual = []
        # check if required columns for creating edges exists
        for property_name, variations in attrs.items():
            actual_name = self.find_actual_column_name(hr_data, variations)
            if actual_name:
                property_to_actual.append(actual_name)
        return property_to_actual

    def preprocess_data(self, df, node_features):
        # Compile a list of all relevant columns from node_features
        relevant_columns = set([col for cols in node_features.values() for col in cols])

        # Initialize LabelEncoder and MinMaxScaler
        le = LabelEncoder()
        scaler = MinMaxScaler()

        for column in relevant_columns:
            # Check if the column exists in the DataFrame to avoid KeyError
            if column in df.columns:
                if df[column].dtype == 'object':
                    # Apply LabelEncoder to categorical columns
                    df[column] = le.fit_transform(df[column].astype(str))
                else:
                    # Apply MinMaxScaler to numerical columns
                    scaler = MinMaxScaler()
                    imputer = SimpleImputer(strategy='constant')
                    df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))

        return df

    # Re-defining the custom function to adapt to the new logic
    def manage_edge_probability(self, edges, hr_data, dept_indices, sub_dept_info=False):
        for i, j in combinations(dept_indices, 2):
            emp_i = hr_data.iloc[i]
            emp_j = hr_data.iloc[j]
            if sub_dept_info:  # When there's detail on 'sub-depart'
                if (emp_i['department'] == emp_j['department']) and (emp_i['sub-depart'] == emp_j['sub-depart']):
                    if np.random.rand() < self.CONNECT_AMONG_SUB_DEPART:  # Same department and 'sub-depart'
                        edges.append([i, j])
                elif emp_i['department'] == emp_j['department']:
                    if np.random.rand() < self.CONNECT_AMONG_DEPARTMENT:  # Same department but not 'sub-depart'
                        edges.append([i, j])
            else:  # Defaults to handling by 'department' with chance
                if emp_i['department'] == emp_j['department']:
                    if np.random.rand() < self.CONNECT_AMONG_DEPARTMENT:
                        edges.append([i, j])

    # if edges are given by user, relationship_data is not None, mapping current users to edges
    def egdes_generator(self, hr_data, given_edges = None):
        edges = []
        mapping_df = self.create_index_id_name_mapping(hr_data)

        if given_edges is not None:
            # mapping name to index and generating edges
            for _, row in given_edges.iterrows():
                source_id = row['source']
                target_id = row['target']
                source_index = mapping_df[mapping_df['id'] == source_id].index.item()
                target_index = mapping_df[mapping_df['id'] == target_id].index.item()

                edges.append([source_index, target_index])
        else:
            # if edges are not given, infer the edges
            if 'sub-depart' in hr_data.columns and hr_data['sub-depart'].notnull().any():
                for dept in hr_data['sub-depart'].unique():
                    dept_indices = hr_data[hr_data['sub-depart'] == dept].index.tolist()
                    self.manage_edge_probability(edges, dept_indices, sub_dept_info=True)
            else:
                for dept in hr_data['department'].unique():
                    dept_indices = hr_data[hr_data['department'] == dept].index.tolist()
                    self.manage_edge_probability(edges, dept_indices, sub_dept_info=False)
        return edges

    def edge_index_generator(self,edges):
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

    def features_generator(self, hr_data, node_features, property_to_actual):
        hr_data_parsed = self.preprocess_data(hr_data, node_features)
        # Exclude 'id' and 'name' columns from features
        feature_columns = [col for col in hr_data_parsed.columns if col in property_to_actual]

        features_data = hr_data[feature_columns].values
        return features_data

    def feature_index_generator(self, features):
        feature_index = torch.tensor(features, dtype=torch.float)
        return feature_index

    def nanCheck(self, hr_data, feature_index):
        # Check for NaN values in features
        if torch.isnan(feature_index).any():
           nan_columns = hr_data.columns[hr_data.isnull().any()].tolist()
           raise ValueError(f"NaN values detected in columns: {', '.join(nan_columns)}")
        return "No NaN values detected."

    def model_training(self, feature_index, edge_index):
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

        # Strategy: Scale and Normalize the Weights
        with torch.no_grad():
            embeddings = model(feature_index, edge_index)
            new_weights = torch.norm(embeddings[edge_index[0]] - embeddings[edge_index[1]], dim=1)

        # Initialize the scaler
        scaler = MinMaxScaler()

        # Reshape new_weights for scaling - sklearn's MinMaxScaler expects a 2D array
        weights_reshaped = new_weights.numpy().reshape(-1, 1)

        # Apply the scaler to the weights
        scaled_weights = scaler.fit_transform(weights_reshaped).flatten()
        
        return scaled_weights

    def data_reshape(self, scaled_weights,edge_index, index_to_name_mapping):
        # Create a DataFrame to export
        edges_with_weights = pd.DataFrame(edge_index.t().numpy(), columns=['source', 'target'])

        # Update the DataFrame with scaled weights
        edges_with_weights['weight'] = scaled_weights

        # Use id to map names
        edges_with_weights['source'] = edges_with_weights['source'].map(index_to_name_mapping)
        edges_with_weights['target'] = edges_with_weights['target'].map(index_to_name_mapping)

        return edges_with_weights