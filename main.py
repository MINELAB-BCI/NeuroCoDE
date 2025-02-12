import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math

##############################################
# Gradient Reversal Layer for Domain Adversarial Training
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
# EEG Encoder with CNN, Spectral & Spatial Attention
##############################################
class EEGEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        """
        EEG trial shape: (batch, 1, T, 5, 5)
        where T is the time dimension and 5x5 is the spatial electrode grid.
        """
        super(EEGEncoder, self).__init__()
        # First conv layer: apply a kernel along the time dimension (e.g., kernel size 65) 
        # with a spatial kernel of 1x1.
        self.conv1 = nn.Conv3d(
            in_channels=1, 
            out_channels=32, 
            kernel_size=(65, 1, 1), 
            stride=(10, 1, 1),
            padding=(32, 0, 0)  # Padding to roughly preserve input dimensions
        )
        self.elu = nn.ELU()
        # Second conv layer: extract spatial information using a 5x5 kernel 
        # while using a kernel size of 1 along the time dimension.
        self.conv2 = nn.Conv3d(
            in_channels=32, 
            out_channels=64, 
            kernel_size=(1, 5, 5), 
            stride=(1, 1, 1),
            padding=(0, 2, 2)
        )
        
        # --- Spectral Attention ---
        # The output of conv2 has shape (batch, 64, T_out, 5, 5). 
        # Assume T_out corresponds to frequency bands.
        # We perform spatial averaging and then generate attention weights per frequency band.
        self.spectral_fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.Sigmoid()
        )
        
        # --- Spatial Attention ---
        # Average over the time dimension and use a 2D convolution to generate a spatial attention map (5x5).
        self.spatial_conv = nn.Conv2d(
            in_channels=64, 
            out_channels=1, 
            kernel_size=3, 
            padding=1
        )
        
        # Final embedding: global average pooling followed by a fully-connected layer.
        self.embedding_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_embedding = nn.Linear(64, embedding_dim)
        
    def forward(self, x):
        """
        x: EEG trial, shape (batch, 1, T, 5, 5)
        """
        # CNN-based feature extraction
        x = self.conv1(x)      # Shape: (batch, 32, T_out, 5, 5)
        x = self.elu(x)
        x = self.conv2(x)      # Shape: (batch, 64, T_out, 5, 5)
        x = self.elu(x)
        
        batch_size, C, D, H, W = x.size()
        
        # --- Spectral Attention ---
        # Average over spatial dimensions -> (batch, 64, D)
        spectral_feat = x.mean(dim=[3, 4])  # Shape: (batch, 64, D)
        # Permute to get each frequency band as a 64-dimensional vector: (batch, D, 64)
        spectral_feat = spectral_feat.permute(0, 2, 1)  # Shape: (batch, D, 64)
        # Compute attention weights for each frequency band
        attn_weights = self.spectral_fc(spectral_feat)  # Shape: (batch, D, 64)
        # Apply the attention weights
        spectral_feat = spectral_feat * attn_weights
        # Permute back to (batch, 64, D)
        spectral_feat = spectral_feat.permute(0, 2, 1)
        # Broadcast the spectral attention weights along spatial dimensions
        spectral_attn = spectral_feat.unsqueeze(-1).unsqueeze(-1)  # Shape: (batch, 64, D, 1, 1)
        x = x * spectral_attn
        
        # --- Spatial Attention ---
        # Average over the time dimension (D) -> (batch, 64, H, W)
        spatial_feat = x.mean(dim=2)
        spatial_attn = self.spatial_conv(spatial_feat)  # Shape: (batch, 1, H, W)
        spatial_attn = torch.sigmoid(spatial_attn)
        # Apply spatial attention by unsqueezing the time dimension: (batch, 1, 1, H, W)
        x = x * spatial_attn.unsqueeze(2)
        
        # Final embedding extraction: global average pooling followed by a fully-connected layer.
        x_pool = self.embedding_pool(x)  # Shape: (batch, 64, 1, 1, 1)
        x_pool = x_pool.view(batch_size, -1)  # Shape: (batch, 64)
        embedding = self.fc_embedding(x_pool)   # Shape: (batch, embedding_dim)
        return embedding

##############################################
# Domain Classifier for Domain Adversarial Training
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
# Relation Network for Few-Shot Metric Learning
##############################################
class RelationNetwork(nn.Module):
    def __init__(self, embed_dim):
        super(RelationNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x1, x2):
        # x1 and x2: shape (batch, embed_dim) or (embed_dim,) for a single sample
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 1:
            x2 = x2.unsqueeze(0)
        combined = torch.cat([x1, x2], dim=-1)
        score = self.fc(combined)
        return score.squeeze(-1)  # Returns (batch,) or a scalar

##############################################
# Message Passing Module for Prototype Refinement
##############################################
class MessagePassing(nn.Module):
    def __init__(self, embed_dim):
        super(MessagePassing, self).__init__()
        # Learnable transformation H(·)
        self.H = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, prototypes, query, num_iterations=1):
        """
        prototypes: tensor of shape (K, embed_dim)
        query: tensor of shape (embed_dim,)
        num_iterations: number of message passing iterations L
        """
        refined = prototypes  # Shape: (K, embed_dim)
        # The query node is used for message passing but is not updated.
        for _ in range(num_iterations):
            # Concatenate prototypes and the query node: shape (K+1, embed_dim)
            V = torch.cat([refined, query.unsqueeze(0)], dim=0)
            new_prototypes = []
            for k in range(refined.size(0)):
                c_k = refined[k]
                message = 0.0
                for m in range(V.size(0)):
                    if m == k:
                        continue
                    # Compute weight: w_{km} = exp(-||c_k - V[m]||^2)
                    weight = torch.exp(-torch.norm(c_k - V[m])**2)
                    message = message + weight * self.H(V[m])
                new_c_k = c_k + message
                new_prototypes.append(new_c_k)
            refined = torch.stack(new_prototypes, dim=0)
        return refined

##############################################
# Complete Few-Shot Model (FRESH)
##############################################
class FRESHModel(nn.Module):
    def __init__(self, embedding_dim=128, num_domains=2):
        """
        embedding_dim: Dimension of the encoder output embedding.
        num_domains: Number of domain classes (e.g., sessions or subjects)
        """
        super(FRESHModel, self).__init__()
        self.encoder = EEGEncoder(embedding_dim=embedding_dim)
        self.domain_classifier = DomainClassifier(embedding_dim, num_domains)
        self.relation_network = RelationNetwork(embedding_dim)
        self.message_passing = MessagePassing(embedding_dim)
        
    def forward(self, x, lambda_grl=1.0):
        """
        x: EEG trial (batch, 1, T, 5, 5)
        lambda_grl: Scaling parameter for the Gradient Reversal Layer
        """
        embedding = self.encoder(x)  # f_θ(x)
        # Apply gradient reversal before passing the embedding to the domain classifier
        domain_logits = self.domain_classifier(grad_reverse(embedding, lambda_grl))
        return embedding, domain_logits

##############################################
# Few-Shot Inference and Loss Calculation Functions
##############################################
def compute_prototypes(embeddings, labels, num_classes):
    """
    Compute each class prototype c_k from the support set embeddings and labels.
    embeddings: Tensor of shape (N, embed_dim)
    labels: Tensor of shape (N,) with values in 0...num_classes-1
    """
    prototypes = []
    for k in range(num_classes):
        class_mask = (labels == k)
        if class_mask.sum() > 0:
            proto = embeddings[class_mask].mean(dim=0)
        else:
            proto = torch.zeros(embeddings.size(1), device=embeddings.device)
        prototypes.append(proto)
    prototypes = torch.stack(prototypes, dim=0)
    return prototypes

def predict(query, prototypes, relation_network):
    """
    Predict the label for a query sample using its relation scores to the prototypes.
    query: Embedding of the query sample, shape (embed_dim,)
    prototypes: Tensor of shape (K, embed_dim)
    relation_network: Module to compute the relation score between two embeddings, g_θ
    """
    scores = []
    for proto in prototypes:
        r = relation_network(query, proto)
        scores.append(r)
    scores = torch.stack(scores)  # Shape: (K,)
    probs = F.softmax(scores, dim=0)
    pred = torch.argmax(probs).item()
    return pred, probs

def metric_loss(embeddings, labels, relation_network):
    """
    Compute the few-shot metric loss (MSE) over all pairs in the support set.
    For each pair, r_{i,j} = g_θ(f_θ(x_i), f_θ(x_j)) and target t_{i,j} is 1 if they belong 
    to the same class, and 0 otherwise.
    embeddings: Tensor of shape (N, embed_dim)
    labels: Tensor of shape (N,)
    """
    N = embeddings.size(0)
    loss = 0.0
    count = 0
    for i in range(N):
        for j in range(N):
            r_ij = relation_network(embeddings[i], embeddings[j])
            t_ij = 1.0 if labels[i] == labels[j] else 0.0
            loss = loss + (r_ij - t_ij) ** 2
            count += 1
    return loss / count

def sparsity_loss(embeddings, lambda_s):
    """
    Compute the sparsity loss: L_sparse = lambda_s * sum(|z_m|).
    embeddings: Tensor of shape (batch, embed_dim)
    """
    return lambda_s * torch.sum(torch.abs(embeddings))

def regularization_loss(encoder, relation_network, x, x_perturbed):
    """
    Compute the regularization loss: L_reg = || r_{i,j}(tilde{x}_i, tilde{x}_j) - r_{i,j}(x_i, x_j) ||^2.
    x, x_perturbed: EEG trials of shape (batch, 1, T, 5, 5)
    """
    emb_orig = encoder(x)           # Shape: (batch, embed_dim)
    emb_pert = encoder(x_perturbed)  # Shape: (batch, embed_dim)
    N = emb_orig.size(0)
    loss = 0.0
    count = 0
    for i in range(N):
        for j in range(N):
            r_orig = relation_network(emb_orig[i], emb_orig[j])
            r_pert = relation_network(emb_pert[i], emb_pert[j])
            loss = loss + (r_orig - r_pert) ** 2
            count += 1
    return loss / count

##############################################
# Example: One Training Step using the FRESH Model
##############################################
if __name__ == '__main__':
    # Hyperparameters
    embedding_dim = 128
    num_domains = 2          # For example, two sessions or subjects
    num_classes = 5          # K-way classification (example)
    n_shot = 5               # Number of samples per class
    lambda_grl = 1.0         # Scaling parameter for GRL
    lambda_sparsity = 1e-4   # Weight for the sparsity loss
    num_msg_iterations = 2   # Number of message passing iterations
    
    # Create model and optimizer
    model = FRESHModel(embedding_dim=embedding_dim, num_domains=num_domains)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    
    # Create dummy data
    # EEG trial shape: (batch, 1, T, 5, 5); here T=100 (sufficient to accommodate conv1 kernel size 65)
    batch_size = 32
    T = 100
    dummy_eeg = torch.randn(batch_size, 1, T, 5, 5)
    # Domain labels (e.g., 0 or 1)
    dummy_domain = torch.randint(0, num_domains, (batch_size,))
    
    # Forward pass: get embeddings and domain predictions
    embeddings, domain_logits = model(dummy_eeg, lambda_grl=lambda_grl)
    
    # Domain classification loss (using CrossEntropy)
    domain_loss = F.cross_entropy(domain_logits, dummy_domain)
    
    # Create a support set for few-shot learning: n_shot * num_classes samples
    support_size = n_shot * num_classes
    support_eeg = torch.randn(support_size, 1, T, 5, 5)
    # Support set labels: n_shot samples for each class
    support_labels = torch.tensor([i for i in range(num_classes) for _ in range(n_shot)])
    support_embeddings = model.encoder(support_eeg)  # f_θ(x)
    
    # Compute prototypes from the support set embeddings
    prototypes = compute_prototypes(support_embeddings, support_labels, num_classes)
    
    # Optionally, refine prototypes using message passing (hierarchical refinement)
    # Here we use the first support sample's embedding as the query (for demonstration)
    query_embedding = support_embeddings[0]
    refined_prototypes = model.message_passing(prototypes, query_embedding, num_iterations=num_msg_iterations)
    
    # Classification prediction for the query sample using the Relation Network
    pred_label, relation_probs = predict(query_embedding, refined_prototypes, model.relation_network)
    print(f"Predicted label: {pred_label}")
    print(f"Relation probabilities: {relation_probs.detach().cpu().numpy()}")
    
    # Compute the metric loss over pairs in the support set (MSE loss)
    met_loss = metric_loss(support_embeddings, support_labels, model.relation_network)
    
    # Compute the sparsity loss on the embeddings
    sp_loss = sparsity_loss(embeddings, lambda_sparsity)
    
    # If perturbed samples are available, compute the regularization loss L_reg
    noise = 0.01 * torch.randn_like(dummy_eeg)
    dummy_eeg_pert = dummy_eeg + noise
    reg_loss = regularization_loss(model.encoder, model.relation_network, dummy_eeg, dummy_eeg_pert)
    
    # Combine all losses (weights for each loss term can be tuned as needed)
    total_loss = domain_loss + met_loss + sp_loss + reg_loss
    print(f"Total Loss: {total_loss.item():.4f}")
    
    # Backpropagation and optimizer step
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print("One training step complete.")
