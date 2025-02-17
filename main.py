import os
import numpy as np
import mne

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function

##############################################
# 1) Define channel order (25 channels → 5x5 grid)
##############################################
channel_order = [
    "F1","F2","F3","F4","F5",
    "F6","FC1","FC2","FC3","FC4",
    "FC5","FC6","C1","C2","C3",
    "C4","C5","C6","CP1","CP2",
    "CP3","CP4","Fz","Cz","CPz"
]

##############################################
# 2) Gradient Reversal Layer
##############################################
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)

##############################################
# 3) EEGEncoder (CNN + Attention)
##############################################
class EEGEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super(EEGEncoder, self).__init__()
        
        # conv1: Temporal axis (kernel_size=(65,1,1)), stride=(10,1,1)
        self.conv1 = nn.Conv3d(
            in_channels=1, 
            out_channels=32, 
            kernel_size=(65, 1, 1), 
            stride=(10, 1, 1),
            padding=(32, 0, 0)
        )
        self.elu = nn.ELU()
        
        # conv2: Spatial axis (kernel_size=(1,5,5))
        self.conv2 = nn.Conv3d(
            in_channels=32, 
            out_channels=64, 
            kernel_size=(1, 5, 5), 
            stride=(1, 1, 1),
            padding=(0, 2, 2)
        )
        
        # Spectral Attention
        self.spectral_fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.Sigmoid()
        )
        
        # Spatial Attention
        self.spatial_conv = nn.Conv2d(
            in_channels=64, 
            out_channels=1, 
            kernel_size=3, 
            padding=1
        )
        
        # Global pooling + FC for embedding
        self.embedding_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_embedding = nn.Linear(64, embedding_dim)
        
    def forward(self, x):
        # x: (batch, 1, T, 5, 5)
        x = self.conv1(x)  # (batch, 32, T', 5, 5)
        x = self.elu(x)
        x = self.conv2(x)  # (batch, 64, T', 5, 5)
        x = self.elu(x)

        batch_size, C, D, H, W = x.size()
        
        # 1) Spectral Attention
        spectral_feat = x.mean(dim=[3,4])  # (batch, 64, D)
        spectral_feat = spectral_feat.permute(0, 2, 1)  # (batch, D, 64)
        attn_weights = self.spectral_fc(spectral_feat)  # (batch, D, 64)
        spectral_feat = spectral_feat * attn_weights
        spectral_feat = spectral_feat.permute(0, 2, 1)  # (batch, 64, D)
        spectral_attn = spectral_feat.unsqueeze(-1).unsqueeze(-1)  # (batch, 64, D, 1, 1)
        x = x * spectral_attn
        
        # 2) Spatial Attention
        spatial_feat = x.mean(dim=2)  # (batch, 64, 5, 5)
        spatial_attn = self.spatial_conv(spatial_feat)  # (batch, 1, 5, 5)
        spatial_attn = torch.sigmoid(spatial_attn)
        x = x * spatial_attn.unsqueeze(2)  # (batch, 64, D, 5, 5)
        
        # 3) Embedding
        x_pool = self.embedding_pool(x)  # (batch, 64, 1, 1, 1)
        x_pool = x_pool.view(batch_size, -1)  # (batch, 64)
        embedding = self.fc_embedding(x_pool) # (batch, embedding_dim)
        return embedding

##############################################
# 4) Domain Classifier
##############################################
class DomainClassifier(nn.Module):
    def __init__(self, embedding_dim, num_domains):
        super(DomainClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ELU(),
            nn.Linear(64, num_domains)
        )
        
    def forward(self, x):
        return self.fc(x)

##############################################
# 5) Relation Network
##############################################
class RelationNetwork(nn.Module):
    def __init__(self, embed_dim):
        super(RelationNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim*2, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x1, x2):
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 1:
            x2 = x2.unsqueeze(0)
        combined = torch.cat([x1, x2], dim=-1)
        score = self.fc(combined)
        return score.squeeze(-1)

##############################################
# 6) Message Passing (Prototype Refinement)
##############################################
class MessagePassing(nn.Module):
    def __init__(self, embed_dim):
        super(MessagePassing, self).__init__()
        self.H = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, prototypes, query, num_iterations=1):
        refined = prototypes
        for _ in range(num_iterations):
            V = torch.cat([refined, query.unsqueeze(0)], dim=0)  # (K+1, embed_dim)
            new_prototypes = []
            for k in range(refined.size(0)):
                c_k = refined[k]
                message = 0.0
                for m in range(V.size(0)):
                    if m == k:
                        continue
                    weight = torch.exp(-torch.norm(c_k - V[m])**2)
                    message += weight * self.H(V[m])
                new_prototypes.append(c_k + message)
            refined = torch.stack(new_prototypes, dim=0)
        return refined

##############################################
# 7) FRESH Model (Encoder + DomainCls + RN + MP)
##############################################
class FRESHModel(nn.Module):
    def __init__(self, embedding_dim=128, num_domains=2):
        super(FRESHModel, self).__init__()
        self.encoder = EEGEncoder(embedding_dim=embedding_dim)
        self.domain_classifier = DomainClassifier(embedding_dim, num_domains)
        self.relation_network = RelationNetwork(embedding_dim)
        self.message_passing = MessagePassing(embedding_dim)
        
    def forward(self, x, lambda_grl=1.0):
        embedding = self.encoder(x)
        domain_logits = self.domain_classifier(grad_reverse(embedding, lambda_grl))
        return embedding, domain_logits

##############################################
# 8) Few-Shot Utility Functions
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
    scores = torch.stack(scores)
    probs = F.softmax(scores, dim=0)
    pred = torch.argmax(probs).item()
    return pred, probs

def metric_loss(embeddings, labels, relation_network):
    N = embeddings.size(0)
    loss = 0.0
    for i in range(N):
        for j in range(N):
            r_ij = relation_network(embeddings[i], embeddings[j])
            t_ij = 1.0 if labels[i] == labels[j] else 0.0
            loss += (r_ij - t_ij)**2
    return loss / (N*N)

def sparsity_loss(embeddings, lambda_s):
    return lambda_s * torch.sum(torch.abs(embeddings))

def regularization_loss(encoder, relation_network, x, x_perturbed):
    emb_orig = encoder(x)
    emb_pert = encoder(x_perturbed)
    N = emb_orig.size(0)
    loss = 0.0
    for i in range(N):
        for j in range(N):
            r_orig = relation_network(emb_orig[i], emb_orig[j])
            r_pert = relation_network(emb_pert[i], emb_pert[j])
            loss += (r_orig - r_pert)**2
    return loss / (N*N)

##############################################
# Additional) Accuracy Evaluation Function
##############################################
from torch.utils.data import DataLoader

def evaluate_accuracy(model, dataset, num_classes=3, batch_size=8):
    """
    Compute embeddings for the entire dataset → calculate class prototypes → predict each sample → accuracy (%)
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_embeddings = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            emb, _ = model(x, lambda_grl=0.0)  # Domain classifier is not used
            all_embeddings.append(emb)
            all_labels.append(y)
    
    all_embeddings = torch.cat(all_embeddings, dim=0)  # (N, embed_dim)
    all_labels = torch.cat(all_labels, dim=0)          # (N,)
    
    # Compute prototypes
    prototypes = compute_prototypes(all_embeddings, all_labels, num_classes)
    
    correct = 0
    total = all_embeddings.size(0)
    for i in range(total):
        q_emb = all_embeddings[i]
        pred_label, _ = predict(q_emb, prototypes, model.relation_network)
        if pred_label == all_labels[i].item():
            correct += 1
    acc = 100.0 * correct / total
    return acc

##############################################
# 9) Dataset (Determine class from file name, divide into chunks)
##############################################
class ThreeClassChunkDataset(Dataset):
    """
    - Multiple BrainVision (.vhdr) files.
    - If the file name contains 'reaching', 'multigrasp', or 'twist', assign corresponding class (0/1/2).
    - Cut the raw data into chunks of chunk_size (in seconds) to form (25, chunk_samples) → (1, chunk_samples, 5,5)
    - Use only the 25 channels (channel_order)
    """
    def __init__(
        self,
        fnames,
        chunk_size=2.0,
        step_size=2.0,
        sfreq_cut=None,
        preload=True
    ):
        super().__init__()
        
        self.data_list = []
        self.label_list = []
        
        for fname in fnames:
            raw = mne.io.read_raw_brainvision(fname, preload=preload)
            
            # Determine label from file name
            f_lower = os.path.basename(fname).lower()
            if 'reaching' in f_lower:
                label = 0
            elif 'multigrasp' in f_lower:
                label = 1
            elif 'twist' in f_lower:
                label = 2
            else:
                print(f"[WARNING] Skipping {fname} as reaching/multigrasp/twist could not be recognized.")
                continue
            
            if sfreq_cut is not None:
                raw.resample(sfreq_cut)
            
            raw.pick_channels(channel_order)
            sfreq = raw.info['sfreq']
            
            chunk_samples = int(chunk_size * sfreq)
            step_samples = int(step_size * sfreq)
            
            data_all = raw.get_data()  # (25, n_times)
            n_times_total = data_all.shape[1]
            
            start = 0
            while start + chunk_samples <= n_times_total:
                segment = data_all[:, start:start+chunk_samples]  # (25, chunk_samples)
                
                # (25, chunk_samples) -> (5,5,chunk_samples) -> (chunk_samples,5,5) -> (1, chunk_samples,5,5)
                seg_reshaped = segment.reshape(5, 5, chunk_samples)
                seg_reshaped = np.transpose(seg_reshaped, (2, 0, 1))
                seg_reshaped = np.expand_dims(seg_reshaped, axis=0)
                
                self.data_list.append(seg_reshaped)
                self.label_list.append(label)
                
                start += step_samples
        
        self.data_list = np.array(self.data_list, dtype=np.float32)
        self.label_list = np.array(self.label_list, dtype=np.int64)
        
        print(f"[INFO] Loaded total {len(self.data_list)} segments.")
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        x = self.data_list[idx]  # (1, chunk_samples, 5,5)
        y = self.label_list[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

##############################################
# 10) Main Execution Example
##############################################
if __name__ == '__main__':
    # (1) List of BrainVision data files (.vhdr)
    fnames = [
        "/path/to/my_reaching_file.vhdr",
        "/path/to/my_multigrasp_file.vhdr",
        "/path/to/my_twist_file.vhdr",
    ]
    
    if len(fnames) == 0:
        print("[ERROR] fnames is empty. Please provide actual paths to .vhdr files.")
        import sys
        sys.exit(0)
    
    # (2) Create Dataset & DataLoader
    dataset = ThreeClassChunkDataset(
        fnames=fnames,
        chunk_size=2.0,
        step_size=2.0,
        sfreq_cut=None,  # If needed, e.g.: sfreq_cut=256
        preload=True
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # (3) Create model
    num_classes = 3
    num_domains = 2  # Assume 2 domains (dummy)
    model = FRESHModel(embedding_dim=128, num_domains=num_domains)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    n_epochs = 2
    
    # (4) Training Loop
    for epoch in range(n_epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(loader):
            # Dummy domain labels
            domain_labels = torch.randint(0, num_domains, (x.size(0),))
            
            # Forward
            embeddings, domain_logits = model(x, lambda_grl=1.0)
            
            # Domain classification loss
            domain_loss = F.cross_entropy(domain_logits, domain_labels)
            
            # Relation Network MSE (same=1, different=0)
            N = embeddings.size(0)
            few_shot_loss = 0.0
            count = 0
            for i in range(N):
                for j in range(N):
                    r_ij = model.relation_network(embeddings[i], embeddings[j])
                    t_ij = 1.0 if y[i] == y[j] else 0.0
                    few_shot_loss += (r_ij - t_ij)**2
                    count += 1
            few_shot_loss /= count
            
            total_loss = domain_loss + few_shot_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if (batch_idx+1) % 5 == 0:
                print(f"[Epoch {epoch+1}] Batch {batch_idx+1} / Loss={total_loss.item():.4f}")
    
    print("Training complete!")
    
    # (5) Evaluate Accuracy
    accuracy = evaluate_accuracy(model, dataset, num_classes=num_classes, batch_size=8)
    print(f"Final Accuracy (on entire dataset) = {accuracy:.2f}%")
