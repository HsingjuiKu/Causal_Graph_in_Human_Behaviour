import os
import hdf5storage
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, GRU, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pgmpy.estimators import PC
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from scipy.stats import entropy
import os
import pandas as pd
from scipy.io import loadmat
import networkx as nx
import matplotlib.pyplot as plt


# Assuming the 'Data' folder is in the current working directory
data_folders = ['./Data']

# Initialize lists to store the separated dataframes
protective_dfs = []
non_protective_dfs = []

# Loop through each data folder
for data_folder in data_folders:
    # List all .mat files in the current data folder
    mat_files = [f for f in os.listdir(data_folder) if f.endswith('.mat')]
    # Load each mat file
    for mat_file in mat_files:
        # Construct the full path to the .mat file
        mat_path = os.path.join(data_folder, mat_file)
        # Load the .mat file
        mat_data = loadmat(mat_path)
        # Convert the data into a pandas dataframe
        df = pd.DataFrame(mat_data['data'])
        # Select only the first 70 columns and the last column (73rd) which contains the behavior label
        df = df.iloc[:, list(range(66)) + [70] + [72]]
        # Split the data based on the protective behavior label
        # Assuming the last column in df is the protective behavior label
        protective_behavior = df.iloc[:, -1]
        protective_df = df[protective_behavior == 1]
        non_protective_df = df[protective_behavior == 0]
        # Append the resulting dataframes to their respective lists
        protective_dfs.append(protective_df)
        non_protective_dfs.append(non_protective_df)

# Concatenate all protective and non-protective dataframes
all_protective_data = pd.concat(protective_dfs, axis=0, ignore_index=True)
all_non_protective_data = pd.concat(non_protective_dfs, axis=0, ignore_index=True)

# Now `all_protective_data` and `all_non_protective_data` hold the protective and non-protective data respectively
# You can process these dataframes as needed for your analysis or save them to new .mat files
print(all_protective_data.shape)
print(all_non_protective_data.shape)


nodes_data = {}

# There are 22 nodes, so we loop through each
for node in range(1, 23):
    # Calculate the index for x, y, and z based on the node number
    x_index = node - 1
    y_index = x_index + 22
    z_index = x_index + 44

    # Extract the data for the current node
    node_data = all_protective_data.iloc[:, [x_index, y_index, z_index]]

    # Assign the node data to the corresponding entry in the dictionary
    nodes_data[f'node_{node}'] = node_data

 # (node_from, node_to) based on the known graph structure
edges = [
    (1, 2), (1, 5), (1, 8), (2, 3), (3, 4), (5, 6), (6, 7),
    (8, 9), (9, 10), (9, 15), (9, 20), (10, 11), (11, 12),
    (12, 13), (13, 14), (15, 16), (16, 17), (17, 18), (18, 19),
    (9, 20), (20, 21), (21, 22)
]

def calculate_kl_divergence(p, q, epsilon=1e-10):
    """计算两个分布之间的KL散度"""
    p_normed = (p + epsilon) / (np.sum(p) + epsilon * len(p))
    q_normed = (q + epsilon) / (np.sum(q) + epsilon * len(q))
    return entropy(p_normed, q_normed)


def analyze_joint_interactions(node_data_1, node_data_2):
    """Analyzes interactions between two nodes and calculates weight based on the direction of influence."""
    directions = []
    kl_divergences = {'1_to_2': [], '2_to_1': []}

    for i in range(3):  # 0 for X, 1 for Y, 2 for Z
        df = pd.DataFrame({
            'node_1': node_data_1[i].iloc[:10000],
            'node_2': node_data_2[i].iloc[:10000]
        })

        pc = PC(df)
        estimated_dag = pc.estimate()
        bn = BayesianModel(estimated_dag.edges())
        bn.fit(df, estimator=MaximumLikelihoodEstimator)

        cpd_node_1 = bn.get_cpds('node_1')
        cpd_node_2 = bn.get_cpds('node_2')

        p_node_1_given_2 = cpd_node_1.values
        p_node_2_given_1 = cpd_node_2.values

        kl_divergence_1_to_2 = np.mean(calculate_kl_divergence(p_node_1_given_2, p_node_2_given_1))
        kl_divergence_2_to_1 = np.mean(calculate_kl_divergence(p_node_2_given_1, p_node_1_given_2))

        if kl_divergence_1_to_2 < kl_divergence_2_to_1:
            directions.append('1_to_2')
        else:
            directions.append('2_to_1')
        kl_divergences['1_to_2'].append(kl_divergence_1_to_2)
        kl_divergences['2_to_1'].append(kl_divergence_2_to_1)

    print(directions)
    # Determine the final direction by majority vote
    final_direction = max(set(directions), key=directions.count)

    # Calculate the final weight as the average of KL divergences in the winning direction
    final_weight = np.mean(kl_divergences[final_direction])

    return final_direction, final_weight



