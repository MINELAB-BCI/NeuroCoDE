import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import mne
import global_config

# Apply initial settings
worker_init = global_config.worker_init
global_generator = global_config._global_generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
subject = "session1_sub2"  # Set subject file name for EEG loading.
subject_dir = "models/sub2"  # Directory for pretrained model.
n_shot = 25
directory_path = f"numpy_save/{subject_dir}/{n_shot}_shot"
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

##############################################
# Real EEG Event Dataset (using MNE)
##############################################
class RealEEGEventDataset(Dataset):
    def __init__(self, tmin=0.0, tmax=1.0):
        super().__init__()
        self.file_paths = {
            "reaching": f"{subject}_reaching_MI.vhdr",
            "multigrasp": f"{subject}_multigrasp_MI.vhdr",
            "twist": f"{subject}_twist_MI.vhdr"
        }
        self.session_triggers = {
            "reaching": [11, 21, 31, 41, 51, 61],
            "multigrasp": [11, 21, 61],
            "twist": [91, 101]
        }
        self.session_labels = {
            "reaching": 0,
            "multigrasp": 1,
            "twist": 2
        }
        X_sessions = {}
        y_sessions = {}
        for session, vhdr_path in self.file_paths.items():
            raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)
            raw.filter(l_freq=8.0, h_freq=13.0, fir_design='firwin')
            events, _ = mne.events_from_annotations(raw)
            triggers = self.session_triggers[session]
            events = np.array([ev for ev in events if ev[2] in triggers])
            events_by_trigger = {code: np.where(events[:,2] == code)[0] for code in triggers}
            local_min = min([len(idxs) for idxs in events_by_trigger.values() if len(idxs) > 0])
            balanced_event_indices = []
            for code, idxs in events_by_trigger.items():
                if len(idxs) >= local_min:
                    perm = np.random.permutation(idxs)
                    balanced_event_indices.extend(perm[:local_min])
            balanced_event_indices = sorted(balanced_event_indices)
            events_balanced = events[balanced_event_indices]
            epochs = mne.Epochs(raw, events_balanced, event_id=None, tmin=tmin, tmax=tmax,
                                baseline=None, preload=True)
            X = epochs.get_data()
            y = epochs.events[:,2]
            y = np.full(len(y), self.session_labels[session])
            X_sessions[session] = X
            y_sessions[session] = y
        self.data_list = np.concatenate(list(X_sessions.values()), axis=0)
        self.label_list = np.concatenate(list(y_sessions.values()), axis=0)
        # Global normalization
        all_data = self.data_list.reshape(-1)
        global_mean = np.mean(all_data)
        global_std = np.std(all_data) + 1e-8
        self.data_list = (self.data_list - global_mean) / global_std

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        x = self.data_list[idx]
        x = x[np.newaxis, :, np.newaxis, :]
        y = self.label_list[idx]
        return torch.from_numpy(x.astype(np.float32)), torch.tensor(y, dtype=torch.long)

##############################################
# Model Definition
##############################################
class EEGEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super(EEGEncoder, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(65,1,1), stride=(10,1,1), padding=(32,0,0))
        self.elu = nn.ELU()
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(1,5,5), stride=(1,1,1), padding=(0,2,2))
        self.spectral_fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.Sigmoid()
        )
        self.spatial_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.embedding_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc_embedding = nn.Linear(64, 128)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.elu(x)
        x = self.conv2(x)
        x = self.elu(x)
        batch_size, C, D, H, W = x.size()
        spectral_feat = x.mean(dim=[3,4]).permute(0,2,1)
        attn_weights = self.spectral_fc(spectral_feat)
        spectral_attn = (spectral_feat * attn_weights).permute(0,2,1).unsqueeze(-1).unsqueeze(-1)
        x = x * spectral_attn
        spatial_attn = torch.sigmoid(self.spatial_conv(x.mean(dim=2))).unsqueeze(2)
        x = x * spatial_attn
        x_pool = self.embedding_pool(x).view(batch_size, -1)
        embedding = self.fc_embedding(x_pool)
        return embedding

class RelationNetwork(nn.Module):
    def __init__(self, embed_dim):
        super(RelationNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim*2, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )
    def forward(self, x1, x2):
        if x1.dim() == 1: x1 = x1.unsqueeze(0)
        if x2.dim() == 1: x2 = x2.unsqueeze(0)
        combined = torch.cat([x1, x2], dim=-1)
        score = self.fc(combined)
        return score.squeeze(-1)

class MessagePassing(nn.Module):
    def __init__(self, embed_dim):
        super(MessagePassing, self).__init__()
        self.H = nn.Linear(embed_dim, embed_dim)
    def forward(self, prototypes, query, num_iterations=1):
        refined = prototypes
        for _ in range(num_iterations):
            V = torch.cat([refined, query.unsqueeze(0)], dim=0)
            new_prototypes = []
            for k in range(refined.size(0)):
                c_k = refined[k]
                message = 0.0
                for m in range(V.size(0)):
                    if m == k: continue
                    weight = torch.exp(-torch.norm(c_k - V[m])**2)
                    message += weight * self.H(V[m])
                new_prototypes.append(c_k + message)
            refined = torch.stack(new_prototypes, dim=0)
        return refined

class FRESHModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super(FRESHModel, self).__init__()
        self.encoder = EEGEncoder(embedding_dim=embedding_dim)
        self.relation_network = RelationNetwork(embedding_dim)
        self.message_passing = MessagePassing(embedding_dim)
    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

class PretrainModel(nn.Module):
    def __init__(self, encoder, num_classes=3):
        super(PretrainModel, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x):
        emb = self.encoder(x)
        logits = self.fc(emb)
        return logits, emb

##############################################
# Utility Functions (compute_prototypes, predict, evaluate_accuracy)
##############################################
def compute_prototypes(embeddings, labels, num_classes):
    prototypes = []
    for k in range(num_classes):
        mask = (labels == k)
        if mask.sum() > 0:
            proto = embeddings[mask].mean(dim=0)
        else:
            proto = torch.zeros(embeddings.size(1), device=embeddings.device)
        prototypes.append(proto)
    return torch.stack(prototypes, dim=0)

def predict(query, prototypes, relation_network):
    scores = []
    for proto in prototypes:
        r = relation_network(query, proto)
        scores.append(r)
    scores = torch.stack(scores).squeeze(-1)
    probs = F.softmax(scores, dim=0)
    pred = torch.argmax(probs).item()
    return pred, probs

def evaluate_accuracy(model, dataset, num_classes=3, batch_size=8):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, worker_init_fn=worker_init, generator=global_generator)
    all_embeddings = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            emb = model(x)
            all_embeddings.append(emb)
            all_labels.append(y)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    prototypes = compute_prototypes(all_embeddings, all_labels, num_classes)
    correct = 0
    total = all_embeddings.size(0)
    successful_trials = []
    for i in range(total):
        q_emb = all_embeddings[i]
        pred_label, _ = predict(q_emb, prototypes, model.relation_network)
        if pred_label == all_labels[i].item():
            correct += 1
            successful_trials.append(i)
    acc = 100.0 * correct / total
    return acc, successful_trials

##############################################
# Load saved information
##############################################
def load_dataset_indices():
    try:
        train_indices = np.load(f"numpy_save/{subject}/{n_shot}_shot/train_indices.npy")
        test_indices = np.load(f"numpy_save/{subject}/{n_shot}_shot/test_indices.npy")
    except Exception as e:
        raise ValueError("Saved Train/Test files not found.")
    return train_indices, test_indices

def index_test_dataset(model, test_dataset):
    final_indices = np.load(f"numpy_save/{subject}/{n_shot}_shot/final_test_indices.npy")
    return final_indices

##############################################
# Main Execution (Second Code Block)
##############################################
if __name__ == '__main__':
    # Load raw EEG data using RealEEGEventDataset
    dataset = RealEEGEventDataset(tmin=0.0, tmax=1.0)
    
    # Load saved Train/Test information
    train_indices, test_indices = load_dataset_indices()
    full_train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    print(f"[INFO] Train samples: {len(full_train_dataset)}, Test samples: {len(test_dataset)}")
    
    few_shot_indices = np.load(f"numpy_save/{subject}/{n_shot}_shot/few_shot_indices.npy")
    few_shot_train_dataset = Subset(full_train_dataset, few_shot_indices)    
    model = FRESHModel(embedding_dim=128).to(device)
    final_model_path = f"numpy_save/{subject}/{n_shot}_shot/final_few_shot_model.pth"
    if os.path.exists(final_model_path):
        model.load_state_dict(torch.load(final_model_path))
    
    final_test_indices = index_test_dataset(model, test_dataset)
    optimizer_finetune = torch.optim.Adam(model.parameters(), lr=1e-5)
    finetune_epochs = 5
    model.train()
    finetune_loader = DataLoader(full_train_dataset, batch_size=len(full_train_dataset),
                                 shuffle=False, num_workers=0, worker_init_fn=worker_init, generator=global_generator)
    for epoch in range(finetune_epochs):
        optimizer_finetune.zero_grad()
        x, y = next(iter(finetune_loader))
        x, y = x.to(device), y.to(device)
        emb_before = model(x).detach()
        loss = ((model(x) - emb_before)**2).mean()
        loss.backward()
        optimizer_finetune.step()
        print(f"[Additional Fine-tuning Epoch {epoch+1}/{finetune_epochs}] Loss: {loss.item():.6f}")
    print("\nFine-tuning complete!")
    
    # Final evaluation
    final_test_dataset = Subset(test_dataset, final_test_indices)
    test_acc, successful_trials = evaluate_accuracy(model, final_test_dataset, num_classes=3, batch_size=8)
    print(f"\nTest Accuracy (n-shot): {test_acc:.2f}%")
    print(f"Successful test trial indices: {successful_trials}")
