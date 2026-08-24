"""
Band-weighted cosine-prototype few-shot pipeline.

Differences from fewshot_main.py:
  1) 25 EEG channels are mapped to a 5x5 grid  -> input shape (1, T, 5, 5)
  2) Two frequency bands are extracted per trial: mu (8-13 Hz) and high-gamma (50-70 Hz)
  3) A LEARNABLE band weight (w_mu, w_hg) fuses the two band embeddings
  4) Classification uses COSINE similarity to class prototypes (no RelationNetwork)

Because the encoder now sees a new input (5x5, 2 bands), the old .pth is not reusable.
This script therefore trains the encoder + band weights from scratch (episodic
prototypical training) and then runs few-shot cosine-prototype evaluation.

Config via environment variables (same style as fewshot_main.py):
  EEG_SUBJECT   e.g. session1_sub1
  SUBJECT_DIR   e.g. .../models/sub1
  N_SHOT        1 / 5 / 25
  TRAIN_EPISODES  number of training episodes (default 200)
  FORCE_TRAIN   "1" to retrain even if a saved model exists
"""
import os
import re
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import mne
import global_config

worker_init = global_config.worker_init
global_generator = global_config._global_generator
SEED = global_config.seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Directory settings (override with env vars; no machine-specific defaults) ----
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_CODE_DIR)
DATA_ROOT = os.environ.get("DATA_ROOT", "")
MODELS_ROOT = os.environ.get("MODELS_ROOT", os.path.join(_REPO_DIR, "weights"))
subject = os.environ.get("EEG_SUBJECT", "session1_sub1")
subject_dir = os.environ.get("SUBJECT_DIR", os.path.join(MODELS_ROOT, "sub1"))
n_shot = int(os.environ.get("N_SHOT", "25"))
directory_path = os.path.join(subject_dir, f"{n_shot}_shot")
# epoch window (seconds relative to cue onset) and model filename, all overridable
TMIN = float(os.environ.get("TMIN", "0.0"))
TMAX = float(os.environ.get("TMAX", "4.0"))
MODEL_FILE = os.environ.get("MODEL_FILE", "band_prototype_model.pth")
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(_REPO_DIR, "cache"))

# ---- 5x5 electrode grid (row-major) ----
CHANNEL_ORDER = [
    "F1", "F2", "F3", "F4", "F5",
    "F6", "FC1", "FC2", "FC3", "FC4",
    "FC5", "FC6", "C1", "C2", "C3",
    "C4", "C5", "C6", "CP1", "CP2",
    "CP3", "CP4", "Fz", "Cz", "CPz",
]
GRID_H, GRID_W = 5, 5

# ---- Frequency bands (recording is lowpass 70 Hz + 60 Hz notch) ----
BANDS = [("mu", 8.0, 13.0), ("hg", 50.0, 70.0)]
NUM_BANDS = len(BANDS)
BAND_NAMES = [b[0] for b in BANDS]

NUM_CLASSES = 3


##############################################
# Dataset: 25ch -> 5x5, two bands, band-wise normalization
##############################################
class BandEEGDataset(Dataset):
    def __init__(self, subject_name=None, tmin=0.0, tmax=1.0):
        super().__init__()
        subject_name = subject_name if subject_name is not None else subject
        self.subject_name = subject_name
        cache_dir = CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        band_tag = "_".join(f"{n}{int(l)}-{int(h)}" for n, l, h in BANDS)
        cache_path = os.path.join(cache_dir, f"{subject_name}_grid5x5_{band_tag}_tmin{tmin}_tmax{tmax}.npz")
        if os.path.exists(cache_path):
            cached = np.load(cache_path)
            self.data_list = cached["data"]      # (N, NUM_BANDS, T, 5, 5)
            self.label_list = cached["labels"]   # (N,)
            print(f"[INFO] Loaded cached dataset: {cache_path}")
            return

        file_paths = {
            "reaching": os.path.join(DATA_ROOT, f"{subject_name}_reaching_MI.vhdr"),
            "multigrasp": os.path.join(DATA_ROOT, f"{subject_name}_multigrasp_MI.vhdr"),
            "twist": os.path.join(DATA_ROOT, f"{subject_name}_twist_MI.vhdr"),
        }
        session_triggers = {
            "reaching": [11, 21, 31, 41, 51, 61],
            "multigrasp": [11, 21, 61],
            "twist": [91, 101],
        }
        session_labels = {"reaching": 0, "multigrasp": 1, "twist": 2}

        X_sessions, y_sessions = {}, {}
        for session, vhdr_path in file_paths.items():
            raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose="ERROR")
            # keep only the 25 grid channels in a fixed order
            raw.pick_channels(CHANNEL_ORDER)
            raw.reorder_channels(CHANNEL_ORDER)

            # events + balanced trial selection computed ONCE (shared across bands)
            events, _ = mne.events_from_annotations(raw, verbose="ERROR")
            triggers = session_triggers[session]
            events = np.array([ev for ev in events if ev[2] in triggers])
            events_by_trigger = {code: np.where(events[:, 2] == code)[0] for code in triggers}
            local_min = min([len(idxs) for idxs in events_by_trigger.values() if len(idxs) > 0])
            balanced_event_indices = []
            for code, idxs in events_by_trigger.items():
                if len(idxs) >= local_min:
                    perm = np.random.permutation(idxs)
                    balanced_event_indices.extend(perm[:local_min])
            balanced_event_indices = sorted(balanced_event_indices)
            events_balanced = events[balanced_event_indices]

            # per-band filtering, identical epoch selection
            band_arrays = []
            for _, l_freq, h_freq in BANDS:
                raw_b = raw.copy().filter(l_freq=l_freq, h_freq=h_freq,
                                          fir_design="firwin", verbose="ERROR")
                epochs = mne.Epochs(raw_b, events_balanced, event_id=None,
                                    tmin=tmin, tmax=tmax, baseline=None,
                                    preload=True, verbose="ERROR")
                Xb = epochs.get_data()  # (n, 25, T)
                band_arrays.append(Xb)

            n_trials = band_arrays[0].shape[0]
            T = band_arrays[0].shape[2]
            # (NUM_BANDS, n, 25, T) -> (n, NUM_BANDS, T, 5, 5)
            stacked = np.stack(band_arrays, axis=0)                 # (B, n, 25, T)
            stacked = stacked.reshape(NUM_BANDS, n_trials, GRID_H, GRID_W, T)
            stacked = stacked.transpose(1, 0, 4, 2, 3)              # (n, B, T, 5, 5)

            y = np.full(n_trials, session_labels[session])
            X_sessions[session] = stacked
            y_sessions[session] = y

        self.data_list = np.concatenate(list(X_sessions.values()), axis=0)
        self.label_list = np.concatenate(list(y_sessions.values()), axis=0)

        # band-wise global normalization (avoid low-freq band dominating high-freq band)
        for b in range(NUM_BANDS):
            band = self.data_list[:, b]
            mean = band.mean()
            std = band.std() + 1e-8
            self.data_list[:, b] = (band - mean) / std

        self.data_list = self.data_list.astype(np.float32)
        np.savez_compressed(cache_path, data=self.data_list, labels=self.label_list)
        print(f"[INFO] Saved dataset cache: {cache_path}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        x = self.data_list[idx]  # (NUM_BANDS, T, 5, 5)
        y = self.label_list[idx]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(y, dtype=torch.long)


##############################################
# Encoder (expects input (B, 1, T, 5, 5))
##############################################
class EEGEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(65, 1, 1), stride=(10, 1, 1), padding=(32, 0, 0))
        self.elu = nn.ELU()
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(1, 5, 5), stride=(1, 1, 1), padding=(0, 2, 2))
        self.spectral_fc = nn.Sequential(
            nn.Linear(64, 32), nn.ELU(), nn.Linear(32, 64), nn.Sigmoid()
        )
        self.spatial_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.embedding_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_embedding = nn.Linear(64, embedding_dim)

    def forward(self, x):
        x = self.elu(self.conv1(x))
        x = self.elu(self.conv2(x))
        b, C, D, H, W = x.size()
        spectral_feat = x.mean(dim=[3, 4]).permute(0, 2, 1)
        attn_weights = self.spectral_fc(spectral_feat)
        spectral_attn = (spectral_feat * attn_weights).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        x = x * spectral_attn
        spatial_attn = torch.sigmoid(self.spatial_conv(x.mean(dim=2))).unsqueeze(2)
        x = x * spatial_attn
        x_pool = self.embedding_pool(x).view(b, -1)
        return self.fc_embedding(x_pool)


##############################################
# Band-weighted cosine-prototype network
##############################################
class BandPrototypeNet(nn.Module):
    def __init__(self, embedding_dim=128, num_bands=NUM_BANDS):
        super().__init__()
        self.encoder = EEGEncoder(embedding_dim=embedding_dim)
        self.num_bands = num_bands
        # learnable band weights (softmax over bands) -> "prototype considers frequency"
        self.band_logits = nn.Parameter(torch.zeros(num_bands))
        # learnable cosine temperature
        self.logit_scale = nn.Parameter(torch.tensor(10.0))
        # supervised pretraining head (used only in phase-1 pretraining)
        self.pretrain_head = nn.Linear(embedding_dim, NUM_CLASSES)

    def band_weights(self):
        return F.softmax(self.band_logits, dim=0)

    def embed(self, x):
        """x: (B, num_bands, T, 5, 5) -> fused, L2-normalized embedding (B, D).

        All bands are processed in a single encoder forward (B*num_bands batch)
        for better GPU utilization.
        """
        B = x.size(0)
        w = self.band_weights()
        xr = x.reshape(B * self.num_bands, 1, *x.shape[2:])   # (B*bands, 1, T, 5, 5)
        e = F.normalize(self.encoder(xr), dim=1)              # (B*bands, D)
        e = e.view(B, self.num_bands, -1)                     # (B, bands, D)
        fused = (w.view(1, self.num_bands, 1) * e).sum(dim=1)  # (B, D)
        return F.normalize(fused, dim=1)

    def classify(self, query_emb, prototypes):
        """cosine logits scaled by temperature."""
        proto = F.normalize(prototypes, dim=1)
        sims = query_emb @ proto.t()               # (Q, num_classes)
        scale = self.logit_scale.clamp(1.0, 100.0)
        return scale * sims


def compute_prototypes(embeddings, labels, num_classes=NUM_CLASSES):
    protos = []
    for k in range(num_classes):
        mask = labels == k
        if mask.sum() > 0:
            protos.append(embeddings[mask].mean(dim=0))
        else:
            protos.append(torch.zeros(embeddings.size(1), device=embeddings.device))
    return torch.stack(protos, dim=0)


##############################################
# Embedding helpers
##############################################
def embed_dataset(model, dataset, batch_size=8):
    pin = device.type == "cuda"
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, worker_init_fn=worker_init, generator=global_generator,
                        pin_memory=pin)
    embs, labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=pin)
            embs.append(model.embed(x))
            labels.append(y.to(device, non_blocking=pin))
    return torch.cat(embs, 0), torch.cat(labels, 0)


##############################################
# Phase 1: supervised pretraining (encoder + band weights init)
##############################################
def pretrain_supervised(model, train_dataset, epochs=40, lr=1e-3, wd=1e-4, batch_size=32):
    pin = device.type == "cuda"
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, worker_init_fn=worker_init, generator=global_generator,
                        pin_memory=pin)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    ce = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        loss_sum, correct, total = 0.0, 0, 0
        for x, y in loader:
            x = x.to(device, non_blocking=pin)
            y = y.to(device, non_blocking=pin)
            emb = model.embed(x)
            logits = model.pretrain_head(emb)
            loss = ce(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            w = model.band_weights().detach().cpu().numpy()
            wtxt = ", ".join(f"{n}={v:.3f}" for n, v in zip(BAND_NAMES, w))
            print(f"[pretrain] epoch {epoch+1:3d}/{epochs}  loss={loss_sum/total:.4f}  "
                  f"acc={100.0*correct/total:5.1f}%  weights[{wtxt}]")
    return model


##############################################
# Phase 2: episodic prototypical fine-tuning
##############################################
def train_episodic(model, train_dataset, num_episodes=2000, n_support=10, n_query=10,
                   lr=3e-4, wd=1e-4):
    # group train indices by class
    labels = np.array([int(train_dataset[i][1]) for i in range(len(train_dataset))])
    by_class = {c: np.where(labels == c)[0] for c in range(NUM_CLASSES)}
    rng = np.random.RandomState(SEED)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_episodes)
    model.train()
    running_acc = None
    for ep in range(num_episodes):
        support_idx, support_lab, query_idx, query_lab = [], [], [], []
        for c in range(NUM_CLASSES):
            idxs = by_class[c].copy()
            if len(idxs) < 2:
                continue
            rng.shuffle(idxs)
            ns = min(n_support, len(idxs) - 1)
            nq = min(n_query, len(idxs) - ns)
            support_idx += list(idxs[:ns]); support_lab += [c] * ns
            query_idx += list(idxs[ns:ns + nq]); query_lab += [c] * nq

        if not support_idx or not query_idx:
            continue

        pin = device.type == "cuda"
        sx = torch.stack([train_dataset[i][0] for i in support_idx]).to(device, non_blocking=pin)
        qx = torch.stack([train_dataset[i][0] for i in query_idx]).to(device, non_blocking=pin)
        s_lab = torch.tensor(support_lab, device=device)
        q_lab = torch.tensor(query_lab, device=device)

        s_emb = model.embed(sx)
        q_emb = model.embed(qx)
        protos = compute_prototypes(s_emb, s_lab, NUM_CLASSES)
        logits = model.classify(q_emb, protos)
        loss = F.cross_entropy(logits, q_lab)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        ep_acc = (logits.argmax(1) == q_lab).float().mean().item() * 100
        running_acc = ep_acc if running_acc is None else 0.98 * running_acc + 0.02 * ep_acc

        if (ep + 1) % 100 == 0 or ep == 0:
            w = model.band_weights().detach().cpu().numpy()
            wtxt = ", ".join(f"{n}={v:.3f}" for n, v in zip(BAND_NAMES, w))
            print(f"[finetune] ep {ep+1:4d}/{num_episodes}  loss={loss.item():.4f}  "
                  f"runacc={running_acc:5.1f}%  weights[{wtxt}]  lr={scheduler.get_last_lr()[0]:.2e}")
    return model


##############################################
# Few-shot cosine-prototype evaluation
##############################################
def evaluate_fewshot(model, support_dataset, query_dataset):
    s_emb, s_lab = embed_dataset(model, support_dataset)
    q_emb, q_lab = embed_dataset(model, query_dataset)
    protos = compute_prototypes(s_emb, s_lab, NUM_CLASSES)
    logits = model.classify(q_emb, protos)
    preds = logits.argmax(1)
    correct = (preds == q_lab)
    acc = 100.0 * correct.float().mean().item()
    # per-class (balanced) accuracy
    per_class = {}
    for c in range(NUM_CLASSES):
        m = q_lab == c
        per_class[c] = 100.0 * correct[m].float().mean().item() if m.sum() > 0 else float("nan")
    balanced_acc = float(np.nanmean(list(per_class.values())))
    successful = [i for i in range(len(correct)) if correct[i].item()]
    return acc, balanced_acc, per_class, successful


def evaluate_fewshot_repeated(model, support_pool, query_dataset, shot_list,
                              n_repeats=100, seed=SEED):
    """Few-shot curve: for each shot, sample support n_repeats times and report mean/std.

    Embeddings are computed ONCE (pool + query); only prototype sampling repeats.
    Support is drawn from the support_pool (target train); query is fixed (held-out test).
    """
    pool_emb, pool_lab = embed_dataset(model, support_pool)
    q_emb, q_lab = embed_dataset(model, query_dataset)
    by_class = {c: np.where(pool_lab.cpu().numpy() == c)[0] for c in range(NUM_CLASSES)}
    results = {}
    with torch.no_grad():
        for n_shot in shot_list:
            rng = np.random.RandomState(seed)
            accs, pcs = [], {c: [] for c in range(NUM_CLASSES)}
            for _ in range(n_repeats):
                protos = []
                for c in range(NUM_CLASSES):
                    idx_c = by_class[c]
                    k = min(n_shot, len(idx_c))
                    sel = rng.choice(idx_c, size=k, replace=False)
                    protos.append(pool_emb[sel].mean(dim=0))
                protos = torch.stack(protos, dim=0)
                preds = model.classify(q_emb, protos).argmax(1)
                correct = (preds == q_lab)
                per = []
                for c in range(NUM_CLASSES):
                    m = q_lab == c
                    v = correct[m].float().mean().item() * 100
                    per.append(v)
                    pcs[c].append(v)
                accs.append(float(np.mean(per)))
            results[n_shot] = (float(np.mean(accs)), float(np.std(accs)),
                               {c: float(np.mean(pcs[c])) for c in range(NUM_CLASSES)})
    return results


def load_dataset_indices():
    train_indices = np.load(os.path.join(directory_path, "train_indices.npy"))
    test_indices = np.load(os.path.join(directory_path, "test_indices.npy"))
    return train_indices, test_indices


def make_balanced_split(labels, seed=SEED, test_ratio=0.2, n_shot=25):
    """Class-balanced, disjoint split returning ABSOLUTE dataset indices.

    - undersample every class to the minimum class count
    - per-class train/test split (test = query, fully balanced)
    - support = n_shot per class drawn from train
    """
    rng = np.random.RandomState(seed)
    per_class = int(min((labels == c).sum() for c in range(NUM_CLASSES)))
    n_test = int(round(per_class * test_ratio))
    train_idx, test_idx, support_idx, query_idx = [], [], [], []
    for c in range(NUM_CLASSES):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        idx = idx[:per_class]
        te = idx[:n_test]
        trn = idx[n_test:]
        sup = trn[:min(n_shot, len(trn))]
        train_idx += trn.tolist()
        test_idx += te.tolist()
        support_idx += sup.tolist()
        query_idx += te.tolist()
    return (np.array(train_idx), np.array(test_idx),
            np.array(support_idx), np.array(query_idx), per_class, n_test)


class ArrayEEGDataset(Dataset):
    """In-memory dataset holding already-preprocessed arrays."""
    def __init__(self, data, labels):
        self.data_list = data
        self.label_list = labels

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        x = self.data_list[idx]
        y = self.label_list[idx]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(y, dtype=torch.long)


def get_other_sessions(subj):
    """Given e.g. 'session1_sub1', return existing other-session names for the same subject."""
    m = re.match(r"session(\d+)_(sub\d+)", subj)
    if not m:
        return []
    sess, who = m.group(1), m.group(2)
    others = []
    for s in ["1", "2", "3"]:
        if s == sess:
            continue
        cand = f"session{s}_{who}"
        if os.path.exists(os.path.join(DATA_ROOT, f"{cand}_reaching_MI.vhdr")):
            others.append(cand)
    return others


def balanced_pool_indices(labels, seed=SEED):
    """Undersample every class to the minimum class count (all kept for training)."""
    rng = np.random.RandomState(seed)
    per_class = int(min((labels == c).sum() for c in range(NUM_CLASSES)))
    idxs = []
    for c in range(NUM_CLASSES):
        i = np.where(labels == c)[0]
        rng.shuffle(i)
        idxs += i[:per_class].tolist()
    return np.array(idxs), per_class


def build_pretrain_dataset(session_names, tmin=0.0, tmax=1.0):
    """Load + concatenate multiple sessions, then class-balance the pool."""
    datas, labels = [], []
    for name in session_names:
        ds = BandEEGDataset(subject_name=name, tmin=tmin, tmax=tmax)
        datas.append(np.asarray(ds.data_list))
        labels.append(np.asarray(ds.label_list))
    data = np.concatenate(datas, axis=0)
    label = np.concatenate(labels, axis=0)
    combined = ArrayEEGDataset(data, label)
    idx, per_class = balanced_pool_indices(label, seed=SEED)
    return Subset(combined, idx), per_class, len(idx)


##############################################
# Main
##############################################
if __name__ == "__main__":
    os.makedirs(directory_path, exist_ok=True)
    if not DATA_ROOT:
        raise SystemExit("Set DATA_ROOT to the GIGA folder that contains session*_sub*_*.vhdr")
    if device.type == "cuda":
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] Using CPU (no CUDA device available)")
    dataset = BandEEGDataset(subject_name=subject, tmin=TMIN, tmax=TMAX)

    balanced = os.environ.get("BALANCED_SPLIT", "1") == "1"
    if balanced:
        labels = np.asarray(dataset.label_list)
        train_indices, test_indices, support_indices, query_indices, per_class, n_test = \
            make_balanced_split(labels, seed=SEED, test_ratio=0.2, n_shot=n_shot)
        full_train_dataset = Subset(dataset, train_indices)
        few_shot_train_dataset = Subset(dataset, support_indices)
        final_test_dataset = Subset(dataset, query_indices)
        # save generated indices (do NOT overwrite the original *_indices.npy)
        for name, arr in [("balanced_train", train_indices), ("balanced_test", test_indices),
                          ("balanced_support", support_indices), ("balanced_query", query_indices)]:
            np.save(os.path.join(directory_path, f"{name}.npy"), arr)
        print(f"[INFO] Balanced split: per_class={per_class} (undersampled), "
              f"train={len(train_indices)} ({per_class-n_test}/class), "
              f"support={len(support_indices)} ({min(n_shot, per_class-n_test)}/class), "
              f"query={len(query_indices)} ({n_test}/class)")
    else:
        train_indices, test_indices = load_dataset_indices()
        full_train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        few_shot_indices = np.load(os.path.join(directory_path, "few_shot_indices.npy"))
        few_shot_train_dataset = Subset(full_train_dataset, few_shot_indices)
        final_test_indices = np.load(os.path.join(directory_path, "final_test_indices.npy"))
        final_test_dataset = Subset(test_dataset, final_test_indices)
        print(f"[INFO] (unbalanced, original indices) Train={len(full_train_dataset)}, "
              f"Test={len(test_dataset)}")

    model = BandPrototypeNet(embedding_dim=128, num_bands=NUM_BANDS).to(device)

    model_path = os.path.join(subject_dir, MODEL_FILE)
    force_train = os.environ.get("FORCE_TRAIN", "0") == "1"
    pretrain_epochs = int(os.environ.get("PRETRAIN_EPOCHS", "40"))
    num_episodes = int(os.environ.get("TRAIN_EPISODES", "2000"))
    # pretraining data source: "other_session" (default) or "self"
    pretrain_data = os.environ.get("PRETRAIN_DATA", "other_session")
    load_only = os.path.exists(model_path) and not force_train

    # build the supervised-pretraining dataset (skipped when only loading a checkpoint)
    if load_only:
        pretrain_dataset = None
    elif pretrain_data == "other_session":
        # PRETRAIN_SESSIONS lets the caller pin the exact pretraining session(s),
        # e.g. "session2_sub3". Falls back to all other sessions if unset.
        explicit = os.environ.get("PRETRAIN_SESSIONS", "").strip()
        if explicit:
            others = [s.strip() for s in explicit.split(",") if s.strip()]
            others = [s for s in others
                      if os.path.exists(os.path.join(DATA_ROOT, f"{s}_reaching_MI.vhdr"))]
        else:
            others = get_other_sessions(subject)
        if others:
            pretrain_dataset, pre_pc, pre_n = build_pretrain_dataset(others, tmin=TMIN, tmax=TMAX)
            print(f"[INFO] Pretrain data = other sessions {others}: "
                  f"{pre_n} trials (balanced {pre_pc}/class)")
        else:
            pretrain_dataset = full_train_dataset
            print("[WARN] No other sessions found; pretraining on target train instead.")
    else:
        pretrain_dataset = full_train_dataset
        print("[INFO] Pretrain data = target train (self)")

    if os.path.exists(model_path) and not force_train:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[INFO] Loaded trained model: {model_path}")
    else:
        print(f"[INFO] Phase 1: supervised pretraining ({pretrain_epochs} epochs)...")
        pretrain_supervised(model, pretrain_dataset, epochs=pretrain_epochs)
        pre_acc, pre_bal, _, _ = evaluate_fewshot(model, few_shot_train_dataset, final_test_dataset)
        print(f"[INFO] After pretraining, test acc = {pre_acc:.2f}% (balanced {pre_bal:.2f}%)")

        print(f"[INFO] Phase 2: episodic prototypical fine-tuning ({num_episodes} episodes)...")
        train_episodic(model, full_train_dataset, num_episodes=num_episodes)
        torch.save(model.state_dict(), model_path)
        print(f"[INFO] Saved model: {model_path}")

    w = model.band_weights().detach().cpu().numpy()
    wtxt = ", ".join(f"{n}={v:.3f}" for n, v in zip(BAND_NAMES, w))
    class_names = {0: "reaching", 1: "multigrasp", 2: "twist"}
    print(f"\nLearned band weights: [{wtxt}]  (temp={model.logit_scale.item():.2f})")

    eval_sweep = os.environ.get("EVAL_SWEEP", "0") == "1"
    if eval_sweep:
        shots = [int(s) for s in os.environ.get("SHOT_LIST", "1,5,25").split(",")]
        reps = int(os.environ.get("EVAL_REPEATS", "100"))
        print(f"[INFO] Repeated few-shot eval: shots={shots}, repeats={reps} "
              f"(support sampled from target train, query=held-out test)")
        res = evaluate_fewshot_repeated(model, full_train_dataset, final_test_dataset, shots, reps)
        print(f"\n{'shot':>5} | {'balanced acc (mean±std)':>24} | reaching | multigrasp | twist")
        print("-" * 78)
        for s in shots:
            m, sd, pc = res[s]
            print(f"{s:>5} | {m:>16.2f} ± {sd:5.2f}% | {pc[0]:7.1f}% | {pc[1]:9.1f}% | {pc[2]:6.1f}%")

        result_json = os.environ.get("RESULT_JSON", "")
        if result_json:
            import json
            out = {
                "subject": subject,
                "band_weights": {n: float(v) for n, v in zip(BAND_NAMES, w)},
                "temp": float(model.logit_scale.item()),
                "repeats": reps,
                "shots": {str(s): {"mean": res[s][0], "std": res[s][1],
                                   "per_class": {class_names[c]: res[s][2][c]
                                                 for c in range(NUM_CLASSES)}}
                          for s in shots},
            }
            with open(result_json, "w") as f:
                json.dump(out, f, indent=2)
            print(f"[INFO] Wrote result json: {result_json}")
    else:
        test_acc, balanced_acc, per_class, successful_trials = evaluate_fewshot(
            model, few_shot_train_dataset, final_test_dataset)
        pctxt = ", ".join(f"{class_names[c]}={per_class[c]:.1f}%" for c in range(NUM_CLASSES))
        print(f"Test Accuracy (n-shot): {test_acc:.2f}%  |  Balanced Accuracy: {balanced_acc:.2f}%")
        print(f"Per-class accuracy: {pctxt}")
        print(f"Successful test trial indices: {successful_trials}")
