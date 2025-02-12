import os
import mne
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Saliency Map을 저장할 디렉토리 설정
saliency_dir = 'saliency_map_trajectory'

# 디렉토리가 없는 경우 생성
os.makedirs(saliency_dir, exist_ok=True)

# 파일 선별 및 처리
data_directory = './GIGA'  # 실제 데이터 파일들이 있는 디렉토리 경로로 변경하세요.
# files = [f for f in os.listdir(data_directory) if 'reaching_MI' in f and f.endswith('.vhdr')]
files = [f for f in os.listdir(data_directory) if f.endswith('.vhdr')]

print(files)

# 주파수 대역 정의 (Delta, Theta, Alpha, Beta, Gamma, High Gamma)
frequency_bands = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 40),
    'High Gamma': (40, 100)  # 하이 감마 대역 추가
}

for file in files[0:]:
    vhdr_file = os.path.join(data_directory, file)
    print("Using@@@@@", vhdr_file)
    print("file num: ", file)
    
    # 1. EEG 데이터 로드
    raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)

        
    motor_channels = ['C3', 'C4', 'Cz', 'F3', 'F4', 'Fz', 'FC1', 'FC2', 'FC3', 'FC4', 'CP1', 'CP2', 'CP3', 'CP4', 'P3', 'P4', 'F1', 'F2']

    # 2. EMG 및 EOG 채널 제외
    # eeg_channels = [ch for ch in raw.ch_names if not (ch.startswith('EMG') or 'EOG' in ch)]
    eeg_channels = [ch for ch in motor_channels if ch in raw.ch_names]
    raw.pick_channels(eeg_channels)

    # 3. 전극 위치 정보 확인 및 설정
    has_montage = raw.get_montage() is not None

    if not has_montage:
        # 전극 위치 정보가 없을 경우 표준 10-20 몽타주 적용
        try:
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage, on_missing='warn')
            print("Standard 10-20 montage applied.")
        except Exception as e:
            print(f"Error applying standard montage: {e}")
            continue

    # 4. 이벤트 추출 및 Epochs 생성
    events, event_id = mne.events_from_annotations(raw)

    event_ids = {'ME1': 11, 'ME2': 21, 'ME3': 31, 'ME4': 41}

    epochs = mne.Epochs(raw, events, event_id=event_ids, tmin=-0.2, tmax=8.0, baseline=None, preload=True)

    # 5. 데이터셋 준비
    X = epochs.get_data()  # (n_epochs, n_channels, n_times)
    y = epochs.events[:, -1]

    label_map = {11: 0, 21: 1, 31: 2, 41: 3}
    y = np.vectorize(label_map.get)(y)

    X = X[:, np.newaxis, :, :]  # (n_epochs, 1, n_channels, n_times)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 6. EEGNet 모델 정의 (전체 채널 사용)
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

    # 7. 모델 학습 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    _, _, channels, samples = X_train.shape
    model = EEGNet(num_classes=4, channels=channels, samples=samples).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 8. 모델 학습
    num_epochs = 150
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    # 9. Saliency Map 생성 및 시각화 (C3 채널에 대해서만)
    saliency_maps = []

    for band_name, (low_freq, high_freq) in frequency_bands.items():
        print(f"Processing frequency band: {band_name}")

        # 주파수 대역별로 데이터 필터링
        raw_filtered = raw.copy().filter(low_freq, high_freq, fir_design='firwin')

        # 필터링된 데이터를 Epochs로 변환
        epochs_filtered = mne.Epochs(raw_filtered, events, event_id=event_ids, tmin=-0.2, tmax=8.0, baseline=None, preload=True)
        X_filtered = epochs_filtered.get_data()[:, np.newaxis, :, :]  # (n_epochs, 1, n_channels, n_times)
        
        # 해당 주파수 대역에서의 y 값을 가져오기
        y_filtered = epochs_filtered.events[:, -1]
        y_filtered = np.vectorize(label_map.get)(y_filtered)

        # 모델을 사용하여 Saliency Map 계산
        def generate_saliency_map(model, data, target_class):
            data.requires_grad = True
            output = model(data)
            model.zero_grad()
            loss = criterion(output, torch.tensor([target_class]).to(device))
            loss.backward()
            saliency = data.grad.abs().detach().cpu().numpy()
            return saliency

        # Saliency Map 계산
        all_saliency_maps = []

        for i in range(len(X_filtered)):
            sample_data = torch.from_numpy(X_filtered[i]).float().unsqueeze(0).to(device)
            sample_label = torch.tensor([y_filtered[i]]).to(device)
            
            saliency_map = generate_saliency_map(model, sample_data, sample_label.item())
            all_saliency_maps.append(saliency_map)

        # numpy 배열로 변환 후 평균 계산
        all_saliency_maps = np.concatenate(all_saliency_maps, axis=0)  # (n_samples, 1, n_channels, n_times)
        mean_saliency_map = np.mean(all_saliency_maps, axis=0).squeeze(0)  # (n_channels, n_times)

        # C3 채널에 해당하는 Saliency Map만 선택
        c3_index = eeg_channels.index('C3')
        c3_saliency_map = mean_saliency_map[c3_index, :]  # (n_times)

        # 주파수 대역별 Saliency Map 저장
        saliency_maps.append(c3_saliency_map)

    # 10. 모든 주파수 대역에 대해 Saliency Map 시각화
    saliency_maps = np.array(saliency_maps)  # (n_bands, n_times)

    times = np.arange(saliency_maps.shape[-1]) / raw.info['sfreq']

    plt.figure(figsize=(10, 2))
    plt.imshow(saliency_maps, aspect='auto', cmap='jet', extent=[times[0], times[-1], 0, len(frequency_bands)], origin='lower')

    plt.yticks(ticks=np.arange(len(frequency_bands)), labels=list(frequency_bands.keys()))
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency Bands')
    plt.title('Saliency Map for C3 Channel Across Frequency Bands')
    plt.colorbar(label='Saliency')

    plt.tight_layout()
    saliency_map_file = os.path.join(saliency_dir, f"{os.path.splitext(file)[0]}_C3_saliency.png")
    plt.savefig(saliency_map_file, dpi=300)
    plt.close()

이것도 있어.
