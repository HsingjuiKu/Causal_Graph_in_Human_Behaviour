import os
import pandas as pd
from scipy.io import loadmat
import hdf5storage
import numpy as np
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

import pandas as pd
from pgmpy.estimators import PC
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator

from scipy.stats import entropy
# Load data

data_folders = ['./Data']

dfs = []
for data_folder in data_folders:
    mat_files = [f for f in os.listdir(data_folder) if f.endswith('.mat')]
    for mat_file in mat_files:
        mat_data = hdf5storage.loadmat(os.path.join(data_folder, mat_file))
        df = pd.DataFrame(mat_data['data'])
        df = df.iloc[:, list(range(70)) + [72]]
        dfs.append(df)
all_data = pd.concat(dfs, axis=0, ignore_index=True)
y = all_data.iloc[:, -1]


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
        df = df.iloc[:, list(range(66)) + [72]]
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


edges = [
    # (node_from, node_to) based on the known graph structure
    (1, 2), (1, 5), (1, 8), (2, 3), (3, 4), (5, 6), (6, 7),
    (8, 9), (9, 10), (9, 15), (9, 20), (10, 11), (11, 12),
    (12, 13), (13, 14), (15, 16), (16, 17), (17, 18), (18, 19),
    (9, 20), (20, 21), (21, 22)
]


node_1_x_data = nodes_data['node_1'][0].iloc[:10000]
node_1_y_data = nodes_data['node_1'][22].iloc[:10000]
node_1_z_data = nodes_data['node_1'][44].iloc[:10000]

node_2_x_data = nodes_data['node_2'][1].iloc[:10000]
node_2_y_data = nodes_data['node_2'][23].iloc[:10000]
node_2_z_data = nodes_data['node_2'][45].iloc[:10000]

df_x = pd.DataFrame({'node_1_x': node_1_x_data, 'node_2_x': node_2_x_data})
df_y = pd.DataFrame({'node_1_y': node_1_y_data, 'node_2_y': node_2_y_data})
df_z = pd.DataFrame({'node_1_z': node_1_z_data, 'node_2_z': node_2_z_data})

# 使用PC算法估计每个坐标维度的DAG结构，以X坐标为例
pc = PC(df_x)
estimated_dag_x = pc.estimate()

# 根据估计的DAG构建贝叶斯网络
bn_x = BayesianModel(estimated_dag_x.edges())

# 对X坐标的贝叶斯网络进行参数学习
bn_x.fit(df_x, estimator=MaximumLikelihoodEstimator)

def calculate_kl_divergence(p, q, epsilon=1e-10):
    p_normed = (p + epsilon) / (np.sum(p) + epsilon * len(p))
    q_normed = (q + epsilon) / (np.sum(q) + epsilon * len(q))
    return entropy(p_normed, q_normed)


cpd_node_1_x = bn_x.get_cpds('node_1_x')
cpd_node_2_x = bn_x.get_cpds('node_2_x')

p_1_given_2 = cpd_node_1_x.values
p_2_given_1 = cpd_node_2_x.values

kl_divergence_1_to_2 = calculate_kl_divergence(p_2_given_1, p_1_given_2)
kl_divergence_2_to_1 = calculate_kl_divergence(p_1_given_2, p_2_given_1)

mean_kl_1_to_2 = np.mean(kl_divergence_1_to_2)
mean_kl_2_to_1 = np.mean(kl_divergence_2_to_1)

print(f"平均KL散度从节点1到节点2: {mean_kl_1_to_2}")
print(f"平均KL散度从节点2到节点1: {mean_kl_2_to_1}")

if mean_kl_1_to_2 > mean_kl_2_to_1:
    print("节点1对节点2的影响更显著。")
else:
    print("节点2对节点1的影响更显著。")

# 使用PC算法估计每个坐标维度的DAG结构，以Y坐标为例
pc_y = PC(df_y)
estimated_dag_y = pc_y.estimate()

# 根据估计的DAG构建贝叶斯网络
bn_y = BayesianModel(estimated_dag_y.edges())

# 对X坐标的贝叶斯网络进行参数学习
bn_y.fit(df_y, estimator=MaximumLikelihoodEstimator)

cpd_node_1_y = bn_y.get_cpds('node_1_y')
cpd_node_2_y = bn_y.get_cpds('node_2_y')

p_1_given_2 = cpd_node_1_y.values
p_2_given_1 = cpd_node_2_y.values

kl_divergence_1_to_2 = calculate_kl_divergence(p_2_given_1, p_1_given_2)
kl_divergence_2_to_1 = calculate_kl_divergence(p_1_given_2, p_2_given_1)

mean_kl_1_to_2 = np.mean(kl_divergence_1_to_2)
mean_kl_2_to_1 = np.mean(kl_divergence_2_to_1)

print(f"Y平均KL散度从节点1到节点2: {mean_kl_1_to_2}")
print(f"Y平均KL散度从节点2到节点1: {mean_kl_2_to_1}")

if mean_kl_1_to_2 > mean_kl_2_to_1:
    print("Y-节点1对节点2的影响更显著。")
else:
    print("Y-节点2对节点1的影响更显著。")

# 使用PC算法估计每个坐标维度的DAG结构，以Y坐标为例
pc_z = PC(df_z)
estimated_dag_z = pc_z.estimate()

# 根据估计的DAG构建贝叶斯网络
bn_z = BayesianModel(estimated_dag_z.edges())

# 对X坐标的贝叶斯网络进行参数学习
bn_z.fit(df_z, estimator=MaximumLikelihoodEstimator)
cpd_node_1_z = bn_z.get_cpds('node_1_z')
cpd_node_2_z = bn_z.get_cpds('node_2_z')

p_1_given_2 = cpd_node_1_z.values
p_2_given_1 = cpd_node_2_z.values

kl_divergence_1_to_2 = calculate_kl_divergence(p_2_given_1, p_1_given_2)
kl_divergence_2_to_1 = calculate_kl_divergence(p_1_given_2, p_2_given_1)

mean_kl_1_to_2 = np.mean(kl_divergence_1_to_2)
mean_kl_2_to_1 = np.mean(kl_divergence_2_to_1)

print(f"Z平均KL散度从节点1到节点2: {mean_kl_1_to_2}")
print(f"Z平均KL散度从节点2到节点1: {mean_kl_2_to_1}")

if mean_kl_1_to_2 > mean_kl_2_to_1:
    print("Z-节点1对节点2的影响更显著。")
else:
    print("Z-节点2对节点1的影响更显著。")