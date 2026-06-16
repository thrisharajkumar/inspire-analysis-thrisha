import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, f1_score
import random
import dnn_mortality_data
import charts


# Step 1: Align time series to a common time grid
def align_time_series(data, time_column='chart_time'):
    all_times = []
    for series in data.values():
        all_times.extend(series[time_column].values)
    common_time = range(min(all_times), max(all_times) + 1)
    df = pd.DataFrame(index=common_time)
    for series_name, series_df in data.items():
        interp_func = interp1d(
            series_df[time_column],
            series_df[series_name],
            kind='linear',
            fill_value='extrapolate'
        )
        df[series_name] = interp_func(common_time)
        df[f'{series_name}_mask'] = np.isin(common_time, series_df[time_column]).astype(float)
    return df


# Step 2: Normalize the data
def normalize_data(df, feature_columns, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        df[feature_columns] = scaler.fit_transform(df[feature_columns])
    else:
        df[feature_columns] = scaler.transform(df[feature_columns])
    return df, scaler


# Step 3: Create sequences for transformer input
def create_sequences(df, seq_length, feature_columns, mask_columns):
    sequences = []
    for i in range(len(df) - seq_length + 1):
        seq = df.iloc[i:i + seq_length][feature_columns + mask_columns].values
        sequences.append(seq)
    return np.array(sequences)


# Step 4: Positional encoding for transformer
def positional_encoding(seq_global_length, model_dim):
    pos = np.arange(seq_global_length)[:, np.newaxis]
    i = np.arange(model_dim)[np.newaxis, :]
    angle_rads = pos / np.power(10000, (2 * (i // 2)) / np.float32(model_dim))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return torch.tensor(angle_rads, dtype=torch.float32)


# Step 5: Dataset for autoencoder (windowed sequences)
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, seq_length, model_dim):
        self.sequences = torch.tensor(np.array(sequences), dtype=torch.float32)
        self.pos_enc = positional_encoding(seq_length, model_dim)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.pos_enc


# Dataset for full subject time series (classification)
class SubjectDataset(Dataset):
    def __init__(self, multi_data, scaler, global_length, feature_columns, mask_columns, d_model):
        self.inputs = []
        self.labels = []
        self.masks = []
        self.pos_enc = positional_encoding(global_length, d_model)
        for subject_info in multi_data.values():
            df = align_time_series(subject_info['timeseries'])
            df, _ = normalize_data(df, feature_columns, scaler)
            input_data = df[feature_columns + mask_columns].values
            if len(input_data) < global_length:
                pad_length = global_length - len(input_data)
                input_data = np.pad(input_data, ((0, pad_length), (0, 0)), mode='constant')
                attention_mask = np.ones(len(df))
                attention_mask = np.pad(attention_mask, (0, pad_length), mode='constant', constant_values=0)
            else:
                attention_mask = np.ones(len(df))
            self.inputs.append(torch.tensor(input_data, dtype=torch.float32))
            self.masks.append(torch.tensor(attention_mask, dtype=torch.float32))
            self.labels.append(subject_info['label'])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.pos_enc, self.labels[idx], self.masks[idx]


# Transformer model
class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_features, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.num_features = num_features
        self.input_projection = nn.Linear(num_features, num_features)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_features,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(num_features, num_features // 2)

        # FIX 1: Deeper classifier head — two layers with ReLU instead of one linear layer
        # A single Linear(8->1) could not find the boundary in the embedding space.
        # Two layers give the classifier more capacity to learn non-linear patterns.
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, src, pos_enc, mask=None, mode='autoencode'):
        proj = self.input_projection(src)
        proj_enc = proj + pos_enc
        enc_out = self.transformer_encoder(proj_enc, src_key_padding_mask=mask)
        if mode == 'autoencode':
            return self.output_projection(enc_out)
        elif mode == 'embed':
            return enc_out.mean(dim=1)
        elif mode == 'classify':
            pooled = enc_out.mean(dim=1)
            return self.classifier(pooled)


# Preprocessing for autoencoding
def preprocess_for_autoencode(subjects_data, seq_length):
    feature_columns = ['glucose', 'potassium', 'sodium', 'creatinine']
    mask_columns = [f'{col}_mask' for col in feature_columns]

    all_dfs = []
    for subject in subjects_data.values():
        time_series = subject['timeseries']
        df = align_time_series(time_series)
        all_dfs.append(df[feature_columns])
    all_data = pd.concat(all_dfs)
    scaler = StandardScaler().fit(all_data)

    all_sequences = []
    for subject in subjects_data.values():
        time_series = subject['timeseries']
        df = align_time_series(time_series)
        df, _ = normalize_data(df, feature_columns, scaler)
        sequences = create_sequences(df, seq_length, feature_columns, mask_columns)
        all_sequences.extend(sequences)

    num_features = len(feature_columns) + len(mask_columns)
    dataset = TimeSeriesDataset(all_sequences, seq_length, num_features)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader, scaler, num_features


# Phase 1: Train autoencoder
def train_autoencoder(dataloader, num_features, device, epochs=10):
    model = TimeSeriesTransformer(
        num_features=num_features,
        nhead=8,
        num_layers=5,
        dim_feedforward=128,
        dropout=0.1
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch, pos_enc in dataloader:
            batch, pos_enc = batch.to(device), pos_enc.to(device)
            optimizer.zero_grad()
            output = model(batch, pos_enc, mode='autoencode')
            target = batch[:, :, :num_features // 2]
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Autoencoder Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

    return model


# Phase 2: Train classifier
def train_classifier(class_dataloader, model, device, epochs=10, pos_weight=1.0):
    model.to(device)

    # FIX 2: Unfreeze the full model — train everything end to end
    # Previously the encoder was frozen after autoencoder pre-training.
    # The autoencoder objective (reconstruct labs) is different from the
    # classification objective (predict mortality). Freezing locked in
    # representations that were not discriminative for death prediction.
    # Now the full network fine-tunes together on the classification task.
    for param in model.parameters():
        param.requires_grad = True

    # FIX 3: Lower learning rate for fine-tuning
    # 0.001 caused the bouncing loss (1.59 -> 1.23 -> 1.30 -> 1.20...).
    # 1e-4 is small enough to fine-tune without destroying the pre-trained weights.
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # FIX 4: Learning rate scheduler — reduces LR when loss plateaus
    # This helps escape the bouncing loss pattern seen in the original output.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch, pos_enc, labels, masks in class_dataloader:
            batch = batch.to(device)
            pos_enc = pos_enc.to(device)
            labels = torch.tensor(labels, dtype=torch.float32).to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            output = model(batch, pos_enc, mask=(masks == 0), mode='classify')
            loss = criterion(output.squeeze(1), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(class_dataloader)
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Classifier Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

    return model


# Train/test split with stratification
def create_train_test_split(subjects_data, train_size=0.7, not_survived_pct=0.5, random_state=42):
    subject_ids = list(subjects_data.keys())
    labels = [subjects_data[pid]['label'] for pid in subject_ids]

    survived_ids = [pid for pid, label in zip(subject_ids, labels) if label == 0]
    not_survived_ids = [pid for pid, label in zip(subject_ids, labels) if label == 1]

    test_size = 1 - train_size
    total_test_subjects = int(len(subject_ids) * test_size)
    test_not_survived = int(total_test_subjects * not_survived_pct)
    test_survived = total_test_subjects - test_not_survived

    if test_not_survived > len(not_survived_ids) or test_survived > len(survived_ids):
        raise ValueError(f"Not enough subjects to achieve desired not_survived_pct ({not_survived_pct:.3f}) in test set.")

    random.seed(random_state)
    test_not_survived_ids = random.sample(not_survived_ids, test_not_survived)
    test_survived_ids = random.sample(survived_ids, test_survived)
    test_ids = test_not_survived_ids + test_survived_ids
    train_ids = [pid for pid in subject_ids if pid not in test_ids]

    train_data = {pid: subjects_data[pid] for pid in train_ids}
    test_data = {pid: subjects_data[pid] for pid in test_ids}

    return train_data, test_data


# Evaluate model — AUROC, AUPRC, F1
def evaluate_model(model, test_dataloader, device):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch, pos_enc, labels, masks in test_dataloader:
            batch = batch.to(device)
            pos_enc = pos_enc.to(device)
            labels = torch.tensor(labels, dtype=torch.float32).to(device)
            masks = masks.to(device)
            output = model(batch, pos_enc, mask=(masks == 0), mode='classify')
            probs = torch.sigmoid(output.squeeze(1)).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    # AUROC
    fpr, tpr, _ = roc_curve(all_labels, all_probs, pos_label=1)
    roc_auc = auc(fpr, tpr)
    print(f"\nAUROC: {roc_auc:.4f}")

    # Best F1 threshold
    best_thresh, best_f1 = 0, 0
    for thresh in np.linspace(0, 1, 101):
        y_pred = (np.array(all_probs) >= thresh).astype(int)
        f1 = f1_score(all_labels, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    print(f"Best F1: {best_f1:.3f} at threshold {best_thresh:.2f}")

    # Class distribution
    num_died = sum(1 for l in all_labels if l == 1)
    num_survived = sum(1 for l in all_labels if l == 0)
    print(f"Test positives (died):    {num_died}")
    print(f"Test negatives (survived): {num_survived}")
    print(f"Death rate in test set:   {num_died/(num_died+num_survived):.3f}")

    # FIX 5: Print probability distribution to diagnose stuck-at-0.47 bug
    prob_array = np.array(all_probs)
    print(f"\nProbability stats:")
    print(f"  Min:  {prob_array.min():.4f}")
    print(f"  Max:  {prob_array.max():.4f}")
    print(f"  Mean: {prob_array.mean():.4f}")
    print(f"  Std:  {prob_array.std():.4f}")
    if prob_array.std() < 0.01:
        print("  WARNING: All probabilities nearly identical — model is not learning to discriminate.")
    else:
        print("  OK: Model is producing varied probabilities across patients.")

    charts.plot_auroc(all_labels, all_probs)
    charts.plot_auprc(all_labels, all_probs)

    return roc_auc


# Extract embeddings for t-SNE visualisation
def extract_embeddings(model, dataloader, device):
    model.eval()
    model.to(device)
    embeddings = []
    labels_list = []

    with torch.no_grad():
        for batch in dataloader:
            src, pos_enc, labels, masks = [item.to(device) for item in batch]
            pooled_emb = model(src, pos_enc, mask=(masks == 0), mode='embed')
            embeddings.append(pooled_emb.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return embeddings, labels


# t-SNE visualisation
def visualise_embeddings(embeddings, labels, title='Before classifier training'):
    perplexity = 30
    if len(labels) < 30:
        perplexity = max(5, int(0.5 * len(labels)))
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate='auto', init='pca', random_state=42)
    embed_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embed_2d[:, 0], embed_2d[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
    plt.colorbar(scatter, ticks=[0, 1], label='Class Label')
    plt.title(f't-SNE Visualization of Time Series Embeddings\n{title}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    filename = f"embeddings_{title.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")


def print_model_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


def get_device():
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return "cuda"
    else:
        print("Using CPU")
        return "cpu"


def main():

    # ---------------------------------------------------------------
    # Data generation — synthetic patients for development
    # Replace generate_dataset() with extract_frames() for real data
    # ---------------------------------------------------------------
    num_subjects = 300
    feature_columns = ['glucose', 'potassium', 'sodium', 'creatinine']
    proportion_died = 0.20
    subjects_data, seq_length = dnn_mortality_data.generate_dataset(
        num_subjects, feature_columns, proportion_died)

    print(f"USING seq_length={seq_length}")

    num_died = sum(1 for s in subjects_data.values() if s['label'] == 1)
    num_survived = sum(1 for s in subjects_data.values() if s['label'] == 0)
    not_survived_pct = num_died / num_survived
    print(f"num_died     = {num_died}")
    print(f"num_survived = {num_survived}")
    print(f"Ratio died     = {not_survived_pct:.3f}")
    print(f"Ratio survived = {1.0 - not_survived_pct:.3f}")

    # ---------------------------------------------------------------
    # Compute global length (max time range across all subjects)
    # ---------------------------------------------------------------
    max_length = 0
    for subject in subjects_data.values():
        df = align_time_series(subject['timeseries'])
        max_length = max(max_length, len(df))
    global_length = max_length

    # ---------------------------------------------------------------
    # Train / test split — stratified by mortality label
    # ---------------------------------------------------------------
    train_data, test_data = create_train_test_split(
        subjects_data, train_size=2/3, not_survived_pct=not_survived_pct)

    # ---------------------------------------------------------------
    # Phase 1: Autoencoder pre-training
    # ---------------------------------------------------------------
    auto_dataloader, scaler, num_features = preprocess_for_autoencode(train_data, seq_length)
    print(f"num_features = {num_features} (4 features + 4 masks)")

    device = get_device()

    print("\n--- Phase 1: Autoencoder pre-training ---")
    model = train_autoencoder(auto_dataloader, num_features, device, epochs=10)

    # ---------------------------------------------------------------
    # Build subject datasets for classification phase
    # ---------------------------------------------------------------
    mask_columns = [f'{col}_mask' for col in feature_columns]
    train_dataset = SubjectDataset(train_data, scaler, global_length, feature_columns, mask_columns, num_features)
    test_dataset  = SubjectDataset(test_data,  scaler, global_length, feature_columns, mask_columns, num_features)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

    # Visualise embeddings BEFORE classifier training
    print("\n--- Extracting embeddings before classifier training ---")
    embeddings, labels = extract_embeddings(model, test_dataloader, device)
    visualise_embeddings(embeddings, labels, title='Before classifier training')

    # ---------------------------------------------------------------
    # Phase 2: Classifier fine-tuning
    # ---------------------------------------------------------------
    num_survived_train = sum(1 for p in train_data.values() if p['label'] == 0)
    num_died_train     = sum(1 for p in train_data.values() if p['label'] == 1)
    pos_weight = num_survived_train / num_died_train
    print(f"\npos_weight = {pos_weight:.2f} (survived/died ratio for class imbalance)")

    print("\n--- Phase 2: Classifier fine-tuning ---")
    print_model_stats(model)
    train_classifier(train_dataloader, model, device, epochs=10, pos_weight=pos_weight)

    # Visualise embeddings AFTER classifier training
    print("\n--- Extracting embeddings after classifier training ---")
    embeddings, labels = extract_embeddings(model, test_dataloader, device)
    visualise_embeddings(embeddings, labels, title='After classifier training')

    # ---------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------
    print("\n--- Evaluation on test set ---")
    evaluate_model(model, test_dataloader, device)


if __name__ == "__main__":
    main()