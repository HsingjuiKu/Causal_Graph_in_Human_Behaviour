import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score
from torch_geometric.nn import TopKPooling, GCNConv
from torch.nn import Linear, ReLU
import torch.nn.functional as F
from imblearn.under_sampling import RandomUnderSampler
import wandb
from scipy.io import loadmat
from torch_geometric.nn import global_mean_pool, global_max_pool



# Set up wandb
wandb.init(project="causkelnet", entity="hsingjui-ku")

# Device configuration
device = torch.device("mps" if torch.mps.is_available() else "cpu")
print(device)
deviceepoch_num = 30  # Training epochs
lr = 0.001      # Learning rate
bs = 64         # Batch size
isCause = False # Whether to include causal relations
dataset_root = '../Data' # Dataset root directory

def get_edge_index_and_edge_attr(ex_type):
    # Your function to generate edge_index based on experiment type (ex_type)
    source_nodes = [0, 1, 0, 4, 0, 7, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 9, 8, 14, 8, 19, 9, 10, 10, 11, 11, 12, 12, 13, 14, 15, 15, 16, 16, 17, 17, 18, 19, 20, 20, 21]
    target_nodes = [1, 0, 4, 0, 7, 0, 2, 1, 3, 2, 5, 4, 6, 5, 8, 7, 9, 8, 14, 8, 19, 8, 10, 9, 11, 10, 12, 11, 13, 12, 15, 14, 16, 15, 17, 16, 18, 17, 20, 19, 21, 20]
    edge_attr = None
    if isCause and ex_type == 3:
        source_nodes = [4, 1, 3, 5, 5, 7, 8, 8, 8, 9, 11, 11, 14, 16, 16, 19, 20]
        target_nodes = [0, 0, 2, 4, 6, 0, 7, 9, 14, 10, 10, 12, 15, 15, 17, 8, 19]
        edge_attr = None
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    return edge_index, edge_attr


# Dataset class
class BinaryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        if isCause:
            return ['cause.dataset']
        else:
            return ['base.dataset',]

    def download(self):
        pass

    def process(self):
        # Processing logic for dataset
        data_folders = [dataset_root]
        protective_dfs = []
        non_protective_dfs = []
        for data_folder in data_folders:
            mat_files = [f for f in os.listdir(data_folder) if f.endswith('.mat')]
            for mat_file in mat_files:
                mat_path = os.path.join(data_folder, mat_file)
                mat_data = loadmat(mat_path)
                df = pd.DataFrame(mat_data['data'])
                df = df.iloc[:, list(range(66)) + [70] + [72]]
                protective_behavior = df.iloc[:, -1]
                protective_df = df[protective_behavior == 1]
                non_protective_df = df[protective_behavior == 0]
                protective_dfs.append(protective_df)
                non_protective_dfs.append(non_protective_df)

        all_protective_data = pd.concat(protective_dfs, axis=0, ignore_index=True)
        all_non_protective_data = pd.concat(non_protective_dfs, axis=0, ignore_index=True)
        all_data = np.concatenate([all_protective_data, all_non_protective_data], axis=0)
        ys = all_data[:, -2:]
        x = all_data[:, np.newaxis, [0, 22, 44]]
        for node in range(2, 23):
            x_index = node - 1
            y_index = x_index + 22
            z_index = x_index + 44
            temp = all_data[:, np.newaxis, [x_index, y_index, z_index]]
            x = np.concatenate([x, temp], axis=1)

        data_list = []
        for i in range(x.shape[0]):
            y = torch.tensor([ys[i, -1]], dtype=torch.float)   
            edge_index, edge_attr = get_edge_index_and_edge_attr(ys[i, -2])
            if ys[i, -2] == 3:
                data = Data(x=torch.tensor(x[i], dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr, y=y)
                data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# Load and split data
def load_data_and_split(dataset_root):
    dataset = BinaryDataset(root=dataset_root)
    dataset = dataset.shuffle()
    num_graphs = len(dataset)
    train_size = int(num_graphs * 0.8)
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, num_graphs))
    train_dataset = dataset.index_select(train_indices)
    test_dataset = dataset.index_select(test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
    
    return train_loader, test_loader

train_loader, test_loader = load_data_and_split(dataset_root)


# Define the model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(3, 64)
        self.pool1 = TopKPooling(64, ratio=0.8)
        self.conv2 = GCNConv(64, 64)
        self.pool2 = TopKPooling(64, ratio=0.8)
        self.lin1 = Linear(64, 64)
        self.lin2 = Linear(64, 32)
        self.lin3 = Linear(32, 1)
        self.act1 = ReLU()
        self.act2 = ReLU()

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x1 = global_mean_pool(x, batch)  # 调用 global_mean_pool
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        x2 = global_mean_pool(x, batch)  # 调用 global_mean_pool
        x = x1 + x2
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = torch.sigmoid(self.lin3(x)).squeeze(1)
        return x



# Training and testing functions
def train(train_loader):
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.y.to(device)
        loss = crit(output, label)
        loss.backward()
        optimizer.step()

def test(loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = (out >= 0.5).int()

            y_true.extend(data.y.clone().detach().tolist())
            y_pred.extend(pred.clone().detach().tolist())

    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, pos_label=1, average='macro')
    f1 = f1_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    return accuracy, macro_precision, f1, recall


# Initialize model, optimizer, and loss function
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
crit = torch.nn.BCELoss()

# Training loop with wandb logging
for epoch in range(1, deviceepoch_num + 1):
    train(train_loader)
    train_accuracy, train_macro_precision, train_f1, train_recall = test(train_loader)
    test_accuracy, test_macro_precision, test_f1, test_recall = test(test_loader)

    # Log metrics to wandb
    wandb.log({
        "Train Accuracy": train_accuracy,
        "Train Macro Precision": train_macro_precision,
        "Train F1": train_f1,
        "Train Recall": train_recall,
        "Test Accuracy": test_accuracy,
        "Test Macro Precision": test_macro_precision,
        "Test F1": test_f1,
        "Test Recall": test_recall
    })
    
    print(f'Epoch: {epoch:03d} -------------------------- '
          f'\nTrain: accuracy: {train_accuracy:.4f}, Macro Precision: {train_macro_precision:.4f}, F1: {train_f1:.4f}, recall: {train_recall:.4f}  '
          f'\nTest : accuracy: {test_accuracy:.4f}, Macro Precision: {test_macro_precision:.4f}, F1: {test_f1:.4f}, recall: {test_recall:.4f}')

# Finish the wandb run
wandb.finish()
