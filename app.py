import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import plotly.graph_objects as go
import copy

st.set_page_config(
    page_title="Federated Learning Simulator",
    page_icon="🔐",
    layout="wide"
)

# Model
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1   = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool    = nn.MaxPool2d(2, 2)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.fc1     = nn.Linear(64 * 7 * 7, 128)
        self.fc2     = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

#Data helpers 
@st.cache_resource
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
    test  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train, test

def split_noniid(dataset, num_clients, alpha):
    labels       = np.array(dataset.targets)
    client_idx   = [[] for _ in range(num_clients)]
    for cls in range(10):
        idx = np.where(labels == cls)[0]
        np.random.shuffle(idx)
        props = np.random.dirichlet(np.repeat(alpha, num_clients))
        props = (props * len(idx)).astype(int)
        props[-1] = len(idx) - props[:-1].sum()
        start = 0
        for cid, cnt in enumerate(props):
            client_idx[cid].extend(idx[start:start+cnt].tolist())
            start += cnt
    return [Subset(dataset, idxs) for idxs in client_idx]

# FL core
def client_train(global_weights, dataset, device, local_epochs, is_byzantine):
    if is_byzantine:
        fake = MNISTNet().to(device)
        for p in fake.parameters():
            nn.init.normal_(p.data, mean=0, std=10.0)
        return fake.state_dict()

    model = MNISTNet().to(device)
    model.load_state_dict(global_weights)
    model.train()
    loader    = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            criterion(model(imgs), lbls).backward()
            optimizer.step()
    return model.state_dict()

def aggregate(global_model, client_weights, defense):
    avg = copy.deepcopy(client_weights[0])
    for key in avg.keys():
        stacked = torch.stack([cw[key].float() for cw in client_weights], dim=0)
        avg[key] = stacked.median(dim=0).values if defense == 'median' else stacked.mean(dim=0)
    global_model.load_state_dict(avg)

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            correct += (model(imgs).argmax(dim=1) == lbls).sum().item()
    return correct / len(test_loader.dataset) * 100

# UI 
st.title("Federated Learning Simulator")
st.markdown("Simulate privacy-preserving distributed training on MNIST — adjust parameters and watch the global model learn.")

st.sidebar.header("Configuration")

num_clients    = st.sidebar.slider("Number of clients",       5, 20, 10)
num_rounds     = st.sidebar.slider("Communication rounds",    3, 15, 8)
local_epochs   = st.sidebar.slider("Local epochs per round",  1,  5, 2)
alpha          = st.sidebar.slider("Non-IID α (lower = more skewed)", 0.1, 5.0, 0.5, step=0.1)
byzantine_frac = st.sidebar.slider("Byzantine client fraction", 0.0, 0.4, 0.1, step=0.1)
defense        = st.sidebar.selectbox("Aggregation defense", ["none (FedAvg)", "median"])

defense_mode    = None if defense == "none (FedAvg)" else "median"
num_byzantine   = int(num_clients * byzantine_frac)
byzantine_ids   = list(range(num_byzantine))

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Byzantine clients:** {num_byzantine} / {num_clients}")
st.sidebar.markdown(f"**Defense:** {defense}")

# Info cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Clients",           num_clients)
col2.metric("Rounds",            num_rounds)
col3.metric("Byzantine",         num_byzantine)
col4.metric("Defense",           defense)

st.markdown("---")

# Distribution preview
with st.expander("📊 View client data distribution"):
    train_data, _ = load_data()
    preview_sets  = split_noniid(train_data, num_clients, alpha)
    labels_arr    = np.array(train_data.targets)
    dist_matrix   = np.zeros((num_clients, 10), dtype=int)
    for i, subset in enumerate(preview_sets):
        counts = np.bincount(labels_arr[subset.indices], minlength=10)
        dist_matrix[i] = counts

    fig_dist = go.Figure(data=go.Heatmap(
        z=dist_matrix,
        x=[str(d) for d in range(10)],
        y=[f"Client {i+1}" for i in range(num_clients)],
        colorscale='Blues',
        text=dist_matrix,
        texttemplate="%{text}",
        showscale=True
    ))
    fig_dist.update_layout(
        title=f"Samples per digit per client (α={alpha})",
        xaxis_title="Digit class",
        yaxis_title="Client",
        height=400
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# Run simulation
if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
    device     = torch.device("cpu")
    train_data, test_data = load_data()
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

    client_datasets = split_noniid(train_data, num_clients, alpha)
    global_model    = MNISTNet().to(device)

    accuracies  = []
    chart_placeholder = st.empty()
    progress_bar      = st.progress(0)
    status            = st.empty()

    for r in range(1, num_rounds + 1):
        status.markdown(f"**Round {r}/{num_rounds}** — clients training locally...")
        global_weights = copy.deepcopy(global_model.state_dict())

        client_weights = [
            client_train(
                global_weights,
                client_datasets[i],
                device,
                local_epochs,
                is_byzantine=(i in byzantine_ids)
            )
            for i in range(num_clients)
        ]

        aggregate(global_model, client_weights, defense_mode)
        acc = evaluate(global_model, test_loader, device)
        accuracies.append(acc)

        # Update live chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, r + 1)),
            y=accuracies,
            mode='lines+markers',
            name='FL Accuracy',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=8)
        ))
        fig.add_hline(y=99.1, line_dash="dash", line_color="green",
                      annotation_text="Centralized baseline (99.1%)")
        fig.update_layout(
            title=f"Global Model Accuracy — Round {r}/{num_rounds}",
            xaxis_title="Communication Round",
            yaxis_title="Test Accuracy (%)",
            yaxis=dict(range=[0, 101]),
            height=450
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        progress_bar.progress(r / num_rounds)

    status.success(f"✅ Done! Final accuracy: {accuracies[-1]:.2f}% | Centralized baseline: 99.1%")

    # Summary
    st.markdown("### Results Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Final Accuracy",      f"{accuracies[-1]:.2f}%")
    c2.metric("Peak Accuracy",        f"{max(accuracies):.2f}%")
    c3.metric("Gap from Centralized", f"{99.1 - accuracies[-1]:.2f}%")