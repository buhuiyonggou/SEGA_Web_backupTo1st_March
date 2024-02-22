import pandas as pd
import numpy as np
from itertools import combinations
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

CONNECT_AMONG_SUB_DEPART = 0.25
CONNECT_AMONG_DEPARTMENT = 0.05
CONNECT_AMONG_ORGANZATION = 0.02

def load_data(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, encoding='utf-8')  # Handles BOM if present
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")

    df.columns = df.columns.str.lower()
    return df

hr_data_path = '/content/4.Human Resource Data.xlsx'
relationship_data_path = '/content/simulated_edges_with_names.csv'

hr_data = load_data(hr_data_path)
hr_data.fillna("missing", inplace=True)

relationship_data = None
if relationship_data_path:
  relationship_data = load_data(relationship_data_path)
  relationship_data.fillna("missing", inplace=True)

column_alignment = {
    'id':['id','emp_id', 'employeeid'],
    'name': ['name', 'full_name', "employee_name", 'first_name'],
    'department': ['department', 'depart', 'sector'],
    'sub-depart':['sub-depart', 'sub-department', 'sub-sector','second-department'],
    'manager': ['manager', 'supervisor', 'manager_name','managername']
}

node_features = {
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

# Function to rename columns based on expected variations
def rename_columns_to_standard(df, column_alignment):
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

hr_data.columns = rename_columns_to_standard(hr_data, column_alignment)

def create_index_id_name_mapping(hr_data):
    index_to_id_name_mapping = [{
        'index': i,
        'id': row['id'],
        'name': row['name'].title()
    } for i, row in hr_data.iterrows()]

    mapping_df = pd.DataFrame(index_to_id_name_mapping)
    return mapping_df

# Function to find the actual column name in the DataFrame based on expected variations
def find_actual_column_name(df, possible_names):
    for actual_name in df.columns.str.lower():
        if actual_name in possible_names:
            return actual_name
    return None

# Mapping from your standardized property names to actual column names in hr_data
property_to_actual = []

for property_name, variations in node_features.items():
    actual_name = find_actual_column_name(hr_data, variations)
    if actual_name:
        property_to_actual.append(actual_name)

# check if required columns for creating edges exists
edge_to_actual = []
for property_name, variations in column_alignment.items():
    actual_name = find_actual_column_name(hr_data, variations)
    if actual_name:
        edge_to_actual.append(actual_name)

def preprocess_data(df, node_features):
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
def manage_edge_probability(edges, dept_indices, sub_dept_info=False):
    for i, j in combinations(dept_indices, 2):
        emp_i = hr_data.iloc[i]
        emp_j = hr_data.iloc[j]
        if sub_dept_info:  # When there's detail on 'sub-depart'
            if (emp_i['department'] == emp_j['department']) and (emp_i['sub-depart'] == emp_j['sub-depart']):
                if np.random.rand() < CONNECT_AMONG_SUB_DEPART:  # Same department and 'sub-depart'
                    edges.append([i, j])
            elif emp_i['department'] == emp_j['department']:
                if np.random.rand() < CONNECT_AMONG_DEPARTMENT:  # Same department but not 'sub-depart'
                    edges.append([i, j])
        else:  # Defaults to handling by 'department' with chance
            if emp_i['department'] == emp_j['department']:
                if np.random.rand() < CONNECT_AMONG_DEPARTMENT:
                    edges.append([i, j])

# This optimized separate handler caters to different department cases, defaulting to 1%
def cross_department_edge_probability(hr_data, employee_indices, probability=CONNECT_AMONG_ORGANZATION):
    num_employees = len(hr_data)
    employee_indices = range(num_employees)
    for i, j in combinations(employee_indices, 2):
        if hr_data.iloc[i]['department'] != hr_data.iloc[j]['department']:
            if np.random.rand() < probability:
                edges.append([i, j])

# if edges are given by user, relationship_data is not None, mapping current users to edges
def egdes_generator(hr_data, given_edges = None):
  edges = []
  mapping_df = create_index_id_name_mapping(hr_data)

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
            manage_edge_probability(edges, dept_indices, sub_dept_info=True)
    else:
        for dept in hr_data['department'].unique():
            dept_indices = hr_data[hr_data['department'] == dept].index.tolist()
            manage_edge_probability(edges, dept_indices, sub_dept_info=False)
  return edges

# Applying the last case for handling employees from different departments
# For simplicity, pass this section by now
# cross_department_edge_probability(employee_indices, 0.05)
edges = egdes_generator(hr_data, relationship_data)

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

len(edges)

hr_data = preprocess_data(hr_data, node_features)

# Exclude 'id' and 'name' columns from features
feature_columns = [col for col in hr_data.columns if col in property_to_actual]

features_data = hr_data[feature_columns].values

feature_index = torch.tensor(features_data, dtype=torch.float)

# Check for NaN values in features
if torch.isnan(feature_index).any():
    print("NaN values detected in features.")
    nan_counts = hr_data.isna().sum()
    print(nan_counts)

# Check for NaN values in edges, normally doesn't happen
if torch.isnan(edge_index).any():
    print("NaN values detected in edge_index.")