import os
import mne
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Directory to save the saliency map
saliency_dir = 'saliency_map_trajectory'
os.makedirs(saliency_dir, exist_ok=True)  # Create directory if it does not exist

# Define the data directory containing EEG files
data_directory = './GIGA'  # Change this path to the actual directory

# Select files with .vhdr extension
files = [f for f in os.listdir(data_directory) if f.endswith('.vhdr')]
print(files)

# Define frequency bands (Delta, Theta, Alpha, Beta, Gamma, High Gamma)
frequency_bands = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 40),
    'High Gamma': (40, 100)  # Added high gamma band
}

for file in files:
    vhdr_file = os.path.join(data_directory, file)
    print("Using@@@@@", vhdr_file)
    print("File number:", file)
    
    # Load EEG data
    raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)

    # Define motor-related EEG channels
    motor_channels = ['C3', 'C4', 'Cz', 'F3', 'F4', 'Fz', 'FC1', 'FC2', 'FC3', 'FC4', 'CP1', 'CP2', 'CP3', 'CP4', 'P3', 'P4', 'F1', 'F2']
    eeg_channels = [ch for ch in motor_channels if ch in raw.ch_names]
    raw.pick_channels(eeg_channels)  # Exclude EMG and EOG channels

    # Check and set electrode montage
    has_montage = raw.get_montage() is not None
    if not has_montage:
        try:
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage, on_missing='warn')
            print("Standard 10-20 montage applied.")
        except Exception as e:
            print(f"Error applying standard montage: {e}")
            continue

    # Extract events and create epochs
    events, event_id = mne.events_from_annotations(raw)
    event_ids = {'ME1': 11, 'ME2': 21, 'ME3': 31, 'ME4': 41}
    epochs = mne.Epochs(raw, events, event_id=event_ids, tmin=-0.2, tmax=8.0, baseline=None, preload=True)

    # Prepare dataset
    X = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    y = epochs.events[:, -1]
    
    # Convert labels to numerical format
    label_map = {11: 0, 21: 1, 31: 2, 41: 3}
    y = np.vectorize(label_map.get)(y)

    X = X[:, np.newaxis, :, :]  # Reshape to (n_epochs, 1, n_channels, n_times)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create PyTorch datasets and dataloaders
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define EEGNet model
    class EEGNet(nn.Module):
        def __init__(self, num_classes=4, channels=60, samples=1000):
            super(EEGNet, self).__init__()
            self.firstconv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
                nn.BatchNorm2d(16)
            )
            self.depthwiseConv = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=(channels, 1), stride=(1, 1), groups=16, bias=False),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
                nn.Dropout(p=0.25)
            )
            self.separableConv = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
                nn.Dropout(p=0.25)
            )
            self.classifier = nn.Linear(32 * ((samples // 32)), num_classes)

        def forward(self, x):
            x = self.firstconv(x)
            x = self.depthwiseConv(x)
            x = self.separableConv(x)
            x = x.flatten(start_dim=1)
            x = self.classifier(x)
            return x

    # Set up training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    _, _, channels, samples = X_train.shape
    model = EEGNet(num_classes=4, channels=channels, samples=samples).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 150
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    # Generate saliency map for C3 channel
    c3_index = eeg_channels.index('C3')
    times = np.arange(X.shape[-1]) / raw.info['sfreq']
    saliency_map = np.random.rand(len(frequency_bands), len(times))  # Placeholder for actual saliency calculation

    # Plot the saliency map
    plt.figure(figsize=(10, 2))
    plt.imshow(saliency_map, aspect='auto', cmap='jet', extent=[times[0], times[-1], 0, len(frequency_bands)], origin='lower')
    plt.yticks(ticks=np.arange(len(frequency_bands)), labels=list(frequency_bands.keys()))
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency Bands')
    plt.title('Saliency Map for C3 Channel')
    plt.colorbar(label='Saliency')
    plt.savefig(os.path.join(saliency_dir, f"{os.path.splitext(file)[0]}_C3_saliency.png"), dpi=300)
    plt.close()
