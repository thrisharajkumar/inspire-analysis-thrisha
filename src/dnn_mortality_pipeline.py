"""
dnn_mortality_pipeline.py

Two-phase DNN mortality prediction pipeline for INSPIRE, now wired to read
real subject JSON files via dnn_mortality_data.load_real_subjects() instead
of synthetic generate_dataset().

Phase 1: unsupervised autoencoder pretraining (no labels)
Phase 2: classifier training on top of the (initially frozen) encoder
"""
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

from sklearn.metrics import roc_curve, auc, f1_score
import random

import dnn_mortality_data
import charts


# ============================================================
# Step 1: Align time series to a common time grid
# ============================================================
def align_time_series(time_series, full_length=None):
    """
    Takes a dict of {feature_name: {chart_time: value, ...}} (the real-data
    shape produced by dnn_mortality_data.load_real_subjects /
    extract_frames_from_json) and aligns every feature onto a common,
    contiguous integer-minute time grid via linear interpolation, adding a
    companion *_mask column (1.0 = observed, 0.0 = interpolated/missing).

    :param time_series: dict[feature_name] -> dict[chart_time] -> value
    :param full_length: if given, the grid is padded/truncated to this many
                         minutes starting at the earliest observed chart_time;
                         if None, the grid spans exactly [min_time, max_time]
                         across all features.
    :return: DataFrame indexed by chart_time, with one column per feature
             plus one `<feature>_mask` column per feature.
    """
    all_times = []
    for chart_time2value in time_series.values():
        all_times.extend(list(chart_time2value.keys()))

    if len(all_times) == 0:
        raise ValueError("align_time_series: no observations in any feature")

    min_time = int(min(all_times))
    max_time = int(max(all_times))

    if full_length is not None:
        common_time = range(min_time, min_time + int(full_length))
    else:
        common_time = range(min_time, max_time + 1)

    df = pd.DataFrame(index=common_time)

    for feature_name, chart_time2value in time_series.items():
        if len(chart_time2value) == 0:
            # No observations at all for this feature -> fill with 0 and
            # mark every point as not-observed. (Pipeline-level callers
            # should ideally impute with a population median instead; 0 is
            # a safe, simple placeholder once features are z-normalised.)
            df[feature_name] = 0.0
            df[f'{feature_name}_mask'] = 0.0
            continue

        if len(chart_time2value) == 1:
            # interp1d needs >= 2 points; duplicate the single point one
            # minute later so interpolation does not produce NaNs.
            (only_t, only_v), = chart_time2value.items()
            chart_time2value = dict(chart_time2value)
            chart_time2value[only_t + 1] = only_v

        x_data = sorted(chart_time2value.keys())
        y_data = [chart_time2value[t] for t in x_data]

        interp_func = interp1d(x_data, y_data, kind='linear', bounds_error=False, fill_value='extrapolate')
        values = interp_func(list(common_time))

        mask_values = [1.0 if t in chart_time2value else 0.0 for t in common_time]

        df[feature_name] = values
        df[f'{feature_name}_mask'] = mask_values

        if df[feature_name].isna().any():
            raise ValueError(f"align_time_series: feature {feature_name} produced NaN values")

    return df


# ============================================================
# Step 2: Normalize the data
# ============================================================
def normalize_data(df, feature_columns, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        df[feature_columns] = scaler.fit_transform(df[feature_columns])
    else:
        df[feature_columns] = scaler.transform(df[feature_columns])
    return df, scaler


# ============================================================
# Step 3: Create sequences for transformer input
# ============================================================
def create_sequences(df, seq_length, feature_columns, mask_columns):
    sequences = []
    for i in range(len(df) - seq_length + 1):
        seq = df.iloc[i:i + seq_length][feature_columns + mask_columns].values
        sequences.append(seq)
    return np.array(sequences)


# ============================================================
# Step 4: Positional encoding for transformer
# ============================================================
def positional_encoding(seq_global_length, model_dim):
    pos = np.arange(seq_global_length)[:, np.newaxis]
    i = np.arange(model_dim)[np.newaxis, :]
    angle_rads = pos / np.power(10000, (2 * (i // 2)) / np.float32(model_dim))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return torch.tensor(angle_rads, dtype=torch.float32)


# ============================================================
# Step 5: PyTorch Datasets
# ============================================================
class TimeSeriesDataset(Dataset):
    """Sliding-window sequences for autoencoder pretraining (no labels)."""
    def __init__(self, sequences, seq_length, model_dim):
        self.sequences = np.array(sequences, dtype=np.float32)
        self.pos_enc = positional_encoding(seq_length, model_dim)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.sequences[idx])
        return seq, self.pos_enc


class SubjectDataset(Dataset):
    """Full-length per-subject series + label, for classification."""
    def __init__(self, multi_data, scaler, global_length, feature_columns, mask_columns, d_model):
        self.inputs = []
        self.labels = []
        self.masks = []
        self.pos_enc = positional_encoding(global_length, d_model)

        for subject_info in multi_data.values():
            df = align_time_series(subject_info['timeseries'], full_length=global_length)
            df, _ = normalize_data(df, feature_columns, scaler)
            input_data = df[feature_columns + mask_columns].values

            if len(input_data) < global_length:
                pad_length = global_length - len(input_data)
                input_data = np.pad(input_data, ((0, pad_length), (0, 0)), mode='constant')
                attention_mask = np.ones(len(df))
                attention_mask = np.pad(attention_mask, (0, pad_length), mode='constant', constant_values=0)
            else:
                input_data = input_data[:global_length]
                attention_mask = np.ones(global_length)

            self.inputs.append(torch.tensor(input_data, dtype=torch.float32))
            self.masks.append(torch.tensor(attention_mask, dtype=torch.float32))
            self.labels.append(subject_info['label'])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.pos_enc, self.labels[idx], self.masks[idx]


# ============================================================
# Plotting helper (unchanged)
# ============================================================
def plot_time_series(df, features):
    plt.figure(figsize=(10, 8))
    for i, feature in enumerate(features, 1):
        plt.subplot(len(features), 1, i)
        interpolated = df[df[f'{feature}_mask'] == 0.0]
        plt.scatter(interpolated.index, interpolated[feature], label=f'{feature} (interpolated)',
                    color='blue', s=50, marker='o', alpha=0.5)
        observed = df[df[f'{feature}_mask'] == 1.0]
        plt.scatter(observed.index, observed[feature], label=f'{feature} (observed)',
                    color='red', s=50, marker='o')
        plt.title(f'{feature} Time Series')
        plt.xlabel('Time')
        plt.ylabel(feature)
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# Transformer model: dual modes (autoencode / embed / classify)
# ============================================================
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
        self.output_projection = nn.Linear(num_features, num_features // 2)  # autoencoding target
        self.classifier = nn.Linear(num_features, 1)  # classification head

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


# ============================================================
# Preprocessing for autoencoding (windowed sequences, all subjects)
# ============================================================
def preprocess_for_autoencode(subjects_data, seq_length, feature_columns):
    mask_columns = [f'{col}_mask' for col in feature_columns]

    all_dfs = []
    for subject in subjects_data.values():
        df = align_time_series(subject['timeseries'])
        all_dfs.append(df[feature_columns])
    all_data = pd.concat(all_dfs)
    scaler = StandardScaler().fit(all_data)

    all_sequences = []
    for subject in subjects_data.values():
        df = align_time_series(subject['timeseries'])
        df, _ = normalize_data(df, feature_columns, scaler)
        if len(df) < seq_length:
            # Subject's observed window is shorter than seq_length -- skip,
            # cannot form even one full window.
            continue
        sequences = create_sequences(df, seq_length, feature_columns, mask_columns)
        all_sequences.extend(sequences)

    if len(all_sequences) == 0:
        raise ValueError(
            "preprocess_for_autoencode: produced 0 sequences -- seq_length "
            f"({seq_length}) may be longer than every subject's observed window. "
            "Try a smaller seq_length or a longer days_before_operation window."
        )

    num_features = len(feature_columns) + len(mask_columns)
    dataset = TimeSeriesDataset(all_sequences, seq_length, num_features)
    dataloader = DataLoader(dataset, batch_size=min(256, len(dataset)), shuffle=True,
                             num_workers=2, pin_memory=True)
    return dataloader, scaler, num_features


# ============================================================
# Training: autoencoder (Phase 1)
# ============================================================
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


# ============================================================
# Training: classifier (Phase 2)
# ============================================================
def train_classifier(class_dataloader, model, device, epochs=10, pos_weight=1.0, freeze_encoder=False):
    """
    :param freeze_encoder: if True, only the classifier head is trained
                            (original behaviour -- known to underperform
                            because the autoencoder objective is not
                            mortality-discriminative). If False (default
                            recommended), the whole network including the
                            encoder is fine-tuned end-to-end on the
                            classification task.
    """
    model.to(device)

    for param in model.transformer_encoder.parameters():
        param.requires_grad = not freeze_encoder
    for param in model.input_projection.parameters():
        param.requires_grad = not freeze_encoder

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device))

    if freeze_encoder:
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001, weight_decay=1e-5)
    else:
        # Lower LR than the frozen-head case: we are now updating the whole
        # network, including pretrained encoder weights, and want to avoid
        # destroying what the autoencoder learned too quickly.
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch, pos_enc, labels, masks in class_dataloader:
            batch = batch.to(device)
            pos_enc = pos_enc.to(device)
            labels = labels.to(device=device, dtype=torch.float32)
            masks = masks.to(device)

            optimizer.zero_grad()
            output = model(batch, pos_enc, mask=(masks == 0), mode='classify')
            loss = criterion(output.squeeze(1), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(class_dataloader)
        print(f"Classifier Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
    return model


# ============================================================
# Train/test split, stratified by outcome
# ============================================================
def create_train_test_split(subjects_data, train_size=0.7, not_survived_pct=0.5, random_state=42):
    """
    Stratified split that is robust to small / heavily imbalanced cohorts
    (e.g. 10 died / 21 survived, or even fewer). Always leaves at least one
    subject of each class in train if that class has >= 2 members, and never
    asks random.sample() for more items than exist in either class.
    """
    subject_ids = list(subjects_data.keys())
    labels = [subjects_data[pid]['label'] for pid in subject_ids]

    survived_ids = [pid for pid, label in zip(subject_ids, labels) if label == 0]
    not_survived_ids = [pid for pid, label in zip(subject_ids, labels) if label == 1]

    test_size = 1 - train_size
    total_test_subjects = max(1, int(len(subject_ids) * test_size))

    desired_test_not_survived = int(round(total_test_subjects * not_survived_pct))
    desired_test_survived = total_test_subjects - desired_test_not_survived

    # Hard cap: never take more than (count - 1) from a class with >= 2
    # members (keep at least 1 for training), and never more than the
    # full count from a class with 0 or 1 members.
    max_test_not_survived = max(0, len(not_survived_ids) - 1) if len(not_survived_ids) >= 2 else len(not_survived_ids)
    max_test_survived = max(0, len(survived_ids) - 1) if len(survived_ids) >= 2 else len(survived_ids)

    test_not_survived = min(desired_test_not_survived, max_test_not_survived, len(not_survived_ids))
    test_survived = min(desired_test_survived, max_test_survived, len(survived_ids))
    test_not_survived = max(0, test_not_survived)
    test_survived = max(0, test_survived)

    random.seed(random_state)
    test_not_survived_ids = random.sample(not_survived_ids, test_not_survived) if test_not_survived > 0 else []
    test_survived_ids = random.sample(survived_ids, test_survived) if test_survived > 0 else []
    test_ids = test_not_survived_ids + test_survived_ids
    train_ids = [pid for pid in subject_ids if pid not in test_ids]

    if len(test_ids) == 0:
        raise ValueError(
            "create_train_test_split: resulting test set is empty -- "
            f"need at least 2 died and 2 survived subjects to form both "
            f"train and test sets (have {len(not_survived_ids)} died, "
            f"{len(survived_ids)} survived)."
        )
    if len(train_ids) == 0:
        raise ValueError("create_train_test_split: resulting train set is empty.")

    train_data = {pid: subjects_data[pid] for pid in train_ids}
    test_data = {pid: subjects_data[pid] for pid in test_ids}

    return train_data, test_data


# ============================================================
# Evaluation
# ============================================================
def evaluate_model(model, test_dataloader, device):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch, pos_enc, labels, masks in test_dataloader:
            batch = batch.to(device)
            pos_enc = pos_enc.to(device)
            labels = labels.to(device=device, dtype=torch.float32)
            masks = masks.to(device)

            output = model(batch, pos_enc, mask=(masks == 0), mode='classify')
            probs = torch.sigmoid(output.squeeze(1)).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    fpr, tpr, _ = roc_curve(all_labels, all_probs, pos_label=1)
    roc_auc = auc(fpr, tpr)

    best_thresh, best_f1 = 0, 0
    for thresh in np.linspace(0, 1, 101):
        y_pred = (all_probs >= thresh).astype(int)
        f1 = f1_score(all_labels, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh

    print(f"AUROC = {roc_auc:.4f}")
    print(f"Best F1 = {best_f1:.3f} at threshold {best_thresh:.2f}")

    num_died = int((all_labels == 1).sum())
    num_survived = int((all_labels == 0).sum())
    print(f"Test actual positive (died): {num_died}")
    print(f"Test actual negative (survived): {num_survived}")
    if num_died + num_survived > 0:
        positive_ratio = num_died / (num_died + num_survived)
        print(f"Test proportion died: {positive_ratio:.3f}, survived: {1.0 - positive_ratio:.3f}")

    charts.plot_auroc(all_labels, all_probs)
    charts.plot_auprc(all_labels, all_probs)

    return roc_auc


# ============================================================
# Embedding extraction + t-SNE visualisation
# ============================================================
def extract_embeddings(model, dataloader, device):
    model.eval()
    model.to(device)
    embeddings = []
    labels_list = []

    with torch.no_grad():
        for batch in dataloader:
            src, pos_enc, labels, masks = [item.to(device) if torch.is_tensor(item) else item for item in batch]
            pooled_emb = model(src, pos_enc, mask=(masks == 0), mode='embed')
            embeddings.append(pooled_emb.cpu().numpy())
            labels_list.append(labels.cpu().numpy() if torch.is_tensor(labels) else np.array(labels))

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return embeddings, labels


def visualise_embeddings(embeddings, labels):
    perplexity = 30
    if len(labels) < 30:
        perplexity = max(1, int(0.5 * len(labels)))
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate='auto', init='pca', random_state=42)
    embed_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embed_2d[:, 0], embed_2d[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
    plt.colorbar(scatter, ticks=[0, 1], label='Class Label')
    plt.title('t-SNE Visualization of Time Series Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig('embeddings.png', dpi=300)


def print_model_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")
    print(f"Total trainable parameters: {trainable_params}")


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ============================================================
# Main
# ============================================================
def main():
    # ---------------------------------------------------------------#
    # CONFIG -- flip USE_REAL_DATA to switch between real JSON subjects
    # and synthetic data without touching anything else below.
    # ---------------------------------------------------------------#
    USE_REAL_DATA = True
    JSON_DIR = "./inspire_subjects_json"   # folder of <subject_id>.json files
    FEATURE_COLUMNS = ['glucose', 'potassium', 'sodium', 'creatinine']
    DAYS_BEFORE_OPERATION = 5

    # Only used when USE_REAL_DATA = False
    NUM_SYNTHETIC_SUBJECTS = 300
    PROPORTION_DIED = 0.20

    # Whether to unfreeze the encoder during classifier training. Strongly
    # recommended True -- the frozen-encoder approach was found to produce
    # a classifier that cannot discriminate between outcomes (loss stuck
    # near random, ~0.47 probability for every patient regardless of label).
    UNFREEZE_ENCODER = True

    AUTOENCODER_EPOCHS = 3
    CLASSIFIER_EPOCHS = 3

    # ---------------------------------------------------------------#
    # Load data
    # ---------------------------------------------------------------#
    if USE_REAL_DATA:
        subjects_data, seq_length = dnn_mortality_data.load_real_subjects(
            JSON_DIR, FEATURE_COLUMNS, days_before_operation=DAYS_BEFORE_OPERATION
        )
    else:
        subjects_data, seq_length = dnn_mortality_data.generate_dataset(
            NUM_SYNTHETIC_SUBJECTS, FEATURE_COLUMNS, PROPORTION_DIED
        )

    num_died = sum(1 for s in subjects_data.values() if s['label'] == 1)
    num_survived = sum(1 for s in subjects_data.values() if s['label'] == 0)
    total = num_died + num_survived
    not_survived_pct = (num_died / total) if total > 0 else 0.5
    print(f"num_died     = {num_died}")
    print(f"num_survived = {num_survived}")
    print(f"Ratio of died     = {not_survived_pct:.3f}")
    print(f"Ratio of survived = {1.0 - not_survived_pct:.3f}")

    # ---------------------------------------------------------------#
    # Determine global full_length (longest aligned series across all subjects)
    # ---------------------------------------------------------------#
    max_length = 0
    for subject in subjects_data.values():
        df = align_time_series(subject['timeseries'])
        max_length = max(max_length, len(df))
    global_length = max_length
    print(f"global_length = {global_length}")

    # ---------------------------------------------------------------#
    # Train/test split
    # ---------------------------------------------------------------#
    train_data, test_data = create_train_test_split(
        subjects_data, train_size=2 / 3, not_survived_pct=not_survived_pct
    )
    print(f"train subjects = {len(train_data)}, test subjects = {len(test_data)}")

    # ---------------------------------------------------------------#
    # Phase 1: autoencoder pretraining
    # ---------------------------------------------------------------#
    auto_dataloader, scaler, num_features = preprocess_for_autoencode(train_data, seq_length, FEATURE_COLUMNS)

    device = get_device()
    print(f"device = {device}")
    print(f"num_features = {num_features} (feature + mask columns)")

    model = train_autoencoder(auto_dataloader, num_features, device, epochs=AUTOENCODER_EPOCHS)

    # ---------------------------------------------------------------#
    # Build classification datasets
    # ---------------------------------------------------------------#
    mask_columns = [f'{col}_mask' for col in FEATURE_COLUMNS]
    train_dataset = SubjectDataset(train_data, scaler, global_length, FEATURE_COLUMNS, mask_columns, num_features)
    test_dataset = SubjectDataset(test_data, scaler, global_length, FEATURE_COLUMNS, mask_columns, num_features)

    train_batch_size = min(32, max(1, len(train_dataset)))
    test_batch_size = min(32, max(1, len(test_dataset)))
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    # ---------------------------------------------------------------#
    # Embeddings + t-SNE visualisation (pre-classifier-training encoder)
    # ---------------------------------------------------------------#
    embeddings, labels = extract_embeddings(model, test_dataloader, device)
    visualise_embeddings(embeddings, labels)

    # ---------------------------------------------------------------#
    # Phase 2: classifier training
    # ---------------------------------------------------------------#
    train_num_survived = sum(1 for p in train_data.values() if p['label'] == 0)
    train_num_died = sum(1 for p in train_data.values() if p['label'] == 1)
    pos_weight = (train_num_survived / train_num_died) if train_num_died > 0 else 1.0
    print(f"pos_weight = {pos_weight:.2f}")

    train_classifier(
        train_dataloader, model, device,
        epochs=CLASSIFIER_EPOCHS, pos_weight=pos_weight,
        freeze_encoder=not UNFREEZE_ENCODER
    )

    print_model_stats(model)

    # ---------------------------------------------------------------#
    # Evaluation
    # ---------------------------------------------------------------#
    evaluate_model(model, test_dataloader, device)


if __name__ == "__main__":
    main()
