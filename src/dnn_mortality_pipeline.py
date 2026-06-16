import pandas as pd
import numpy as np
from scapy.layers.tls.crypto.groups import modp2048
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# New imports needed for train-test split and metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import random
import dnn_mortality_data
import charts


from sklearn.metrics import f1_score

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
    """
    Generates fixed length (seq_length) sequences from the feature_columns in df.
    If a value was masked in df, it is also masked in the returned np.array.
    :param df:
    :param seq_length:
    :param feature_columns:
    :param mask_columns:
    :return:
    """
    sequences = []
    for i in range(len(df) - seq_length + 1):
        seq = df.iloc[i:i + seq_length][feature_columns + mask_columns].values
        sequences.append(seq)
    return np.array(sequences)


# Step 4: Positional encoding for transformer
def positional_encoding(seq_global_length, model_dim):
    """
    Generate a sequence (length seq_length for embedding, or global for classification) of positions.
    :param seq_global_length:
    :param model_dim:
    :return:
    """
    pos = np.arange(seq_global_length)[:, np.newaxis]
    i = np.arange(model_dim)[np.newaxis, :]
    angle_rads = pos / np.power(10000, (2 * (i // 2)) / np.float32(model_dim))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return torch.tensor(angle_rads, dtype=torch.float32)


# Step 5: Custom PyTorch Dataset for sequences (autoencoding)
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, seq_length, model_dim):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.pos_enc = positional_encoding(seq_length, model_dim)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.pos_enc


# Custom Dataset for full subject time series (classification)
class SubjectDataset(Dataset):
    def __init__(self, multi_data, scaler, global_length, feature_columns, mask_columns, d_model):
        self.inputs = []
        self.labels = []
        self.masks = []  # Attention masks for padded regions
        self.pos_enc = positional_encoding(global_length, d_model)
        for subject_info in multi_data.values():
            df = align_time_series(subject_info['timeseries'])
            df, _ = normalize_data(df, feature_columns, scaler)
            input_data = df[feature_columns + mask_columns].values
            # Pad if shorter than full_length
            if len(input_data) < global_length:
                pad_length = global_length - len(input_data)
                input_data = np.pad(input_data, ((0, pad_length), (0, 0)), mode='constant')
                # Create attention mask: 1 for valid, 0 for padded
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


# Plot time series (unchanged)
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


# Transformer Model with dual modes: autoencode and classify
# NB: This "model" is used for both "autoencoding" and "classification"!!!
#     I think this is a bad idea that needs to be rectified, confusing to say the least.
#     particularly when we think about freezing learn "autoencoding" weights but
#     still learn weights to classify.
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
        self.output_projection = nn.Linear(num_features, num_features // 2)  # For autoencoding
        self.classifier = nn.Linear(num_features, 1)  # For classification

    def forward(self, src, pos_enc, mask=None, mode='autoencode'):
        """
        Input from PyTorch DataLoader (dataloader) for dataset, where each batch
        yields (src, pos_enc, mask, labels).
        :param src: is the input time series (shape: batch_size × seq_len × num_features)
        :param pos_enc: is positional encoding (same shape as src)
        :param mask: is the padding mask (if applicable)
        :param mode: binary class labels (shape: batch_size)
        :return:
        """
        # Linear transformation of input, mixing timer series with their missing values masks
        proj = self.input_projection(src)
        # Add positional information (not learned but used to learn)
        # Purpose: Adds temporal information, ensuring the transformer distinguishes the order
        # of the 50 time steps,critical since the 4 time series and 4 masks evolve over time.
        # Not an append, adds each element-wise, (32,50,8)+(32,50,8)=(32,50,8)
        proj_enc = proj + pos_enc
        # enc_out is same dims as src but these are "contextual representations"
        # src_key_padding_mask is only for ignoring entire time steps
        #   individual missing/padded feature values are handled by feature masks, not global masks
        # NB: Need to give name / denote these two types of masks!!!
        enc_out = self.transformer_encoder(proj_enc, src_key_padding_mask=mask)
        # enc_out (shape: batch_size × seq_len × num_features)
        if mode == 'autoencode':
            # This is not the classification embedding but the autoencoder’s attempt to reconstruct
            # a lower-dimensional representation of the input.  Like a projection head in contrastive learning.
            # Also, the shape is [32, 24, 8] -> [32, 24, 4]
            # An embedding would be [32, 24, 8] -> [32, 8], where the embedding is vector of length 8
            enc = self.output_projection(enc_out)
            # print(f"TimeSeriesTransformer autoencode:  src={src.shape}  mask={mask}  enc_out={enc_out.shape}  enc={enc.shape}")
            # batch = 23, seq_length = 24, num_features = 8
            # src=torch.Size([32, 24, 8])  mask=None  enc_out=torch.Size([32, 24, 8])  enc=torch.Size([32, 24, 4])
            # enc(shape: batch_size × seq_len × num_features//2)
            return enc
            # return self.output_projection(enc_out)
        elif mode == 'embed':
            pooled = enc_out.mean(dim=1)
            return pooled
        elif mode == 'classify':
            # Get the embedding (i.e. mean pooling over time steps)
            pooled = enc_out.mean(dim=1)
            # Using the "learned embedding", now learn how to classify
            # This is a very shallow (basically logistic regression) classification network
            return self.classifier(pooled)


# Preprocessing for autoencoding (windowed sequences from all subjects)
# Very important!!! The models expect features first then masks
# The multi_data has them interspersed, this function rearranges so features first then masks.
# This function does quite a lot, aligns + determines masks, normalises, then sequences.
#
# My thought was to save the result, however the original multi_data is quite compact
# The preprocessing expands each time series to full_length, adds masks (fairly minor),
# and adds sequences (which greatly increase the size).
def preprocess_for_autoencode(subjects_data, seq_length):
    """
    First aligns the time series in multi_data so all the same full_length with masks
    for observed and observed values.  Then takes these longer series and breaks
    each (including the masks) into sub-sequences of seq_length.
    :param multi_data:
    :param seq_length: length of sub-sequences
    :param full_length:
    :return: dataloader, scaler, num_features (num time series + num masks time series)
    """
    feature_columns = ['glucose', 'potassium', 'sodium', 'creatinine']
    mask_columns = [f'{col}_mask' for col in feature_columns]

    # Collect all data for fitting scaler
    all_dfs = []
    for subject in subjects_data.values():
        time_series = subject['timeseries']
        df = align_time_series(time_series)
        all_dfs.append(df[feature_columns])
    all_data = pd.concat(all_dfs)
    scaler = StandardScaler().fit(all_data)

    # Process each subject
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


# Training function for autoencoding
def train_autoencoder(dataloader, num_features, device, epochs=10):
    """
    Creates and trains a model that jointly learns to encode the time series data (in dataloader).
    :type epochs: object
    :param dataloader: produces n time series features and n masks for those series
    :param num_features: 2*n (total number of columns, first n are time series values, next n are the masks)
    :param device: str to use for matrix computation
    :param epochs:
    :return: trained encoder model
    """

    """
    Are the Features Learned Jointly or Separately?
    The features (glucose, potassium, sodium, creatinine) are learned jointly in
    the TimeSeriesTransformer. Joint Learning Implications:
    
        The transformer learns a shared representation that captures temporal
        patterns and inter-feature relationships (e.g., correlations between
        glucose and sodium levels). This is powerful for modeling multivariate
        time series where features may influence each other.
        
        The masks (indicating observed vs. interpolated values) are included
        in the input, allowing the model to potentially weigh observed data
        more heavily or learn patterns related to missingness, but these
        are also processed jointly with the feature values.
    """


    # The input to the transformer consists of sequences of length seq_length
    # with d_model dimensions, where d_model is the total number of features
    # (4 feature columns: glucose, potassium, sodium, creatinine) plus their
    # corresponding masks (4 mask columns, indicating observed or
    # interpolated values). Thus, d_model = 8 (4 features + 4 masks).

    # model = TimeSeriesTransformer(
    #     num_features=num_features,
    #     nhead=4,
    #     num_layers=1,
    #     dim_feedforward=64,
    #     dropout=0.1
    # ).to(device)
    model = TimeSeriesTransformer(
        num_features=num_features,
        nhead=8, # 4 or 8
        num_layers=5, # 3 or 5
        dim_feedforward=128,
        dropout=0.1
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # The output of the transformer, after passing through the output_projection
    # layer, has d_model // 2 = 4 dimensions, corresponding to the 4 feature
    # columns (glucose, potassium, sodium, creatinine).

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch, pos_enc in dataloader:
            batch, pos_enc = batch.to(device), pos_enc.to(device)
            optimizer.zero_grad()
            output = model(batch, pos_enc, mode='autoencode')
            # The target for training is batch[:, :, :d_model//2], which extracts
            # the feature columns (not the masks) from the input batch. This means
            # the model is trained to reconstruct the input sequence of feature values,
            # not to predict the next value in the time series.
            target = batch[:, :, :num_features // 2]  # Feature columns
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Autoencoder Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

    return model


# Training function for classification with frozen encoder
def train_classifier(class_dataloader, model, device, epochs=10, pos_weight=1.0):

    model.to(device)

    # Model has two parts, transformer_encoder and input_projection
    # Respectively the autoencoder (transformer_encoder) to encode time series
    # and a classification task (input_projection)
    # Freeze autoencoder weights, cannot be changed.
    for param in model.transformer_encoder.parameters():
        param.requires_grad = False
    for param in model.input_projection.parameters():
        param.requires_grad = False

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001, weight_decay=1e-5)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch, pos_enc, labels, masks in class_dataloader:
            batch, pos_enc, labels, masks = batch.to(device), pos_enc.to(device), torch.tensor(labels, dtype=torch.float32).to(device), masks.to(device)
            optimizer.zero_grad()
            output = model(batch, pos_enc, mask=(masks == 0), mode='classify')  # Mask padded regions
            loss = criterion(output.squeeze(1), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(class_dataloader)
        print(f"Classifier Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    return model


# New function to split subjects into train and test sets with balanced 'not survived' percentage
def create_train_test_split(subjects_data, train_size=0.7, not_survived_pct=0.5, random_state=42):
    # Extract subject IDs and labels
    subject_ids = list(subjects_data.keys())
    labels = [subjects_data[pid]['label'] for pid in subject_ids]

    # Separate survived (0) and not survived (1) subjects
    survived_ids = [pid for pid, label in zip(subject_ids, labels) if label == 0]
    not_survived_ids = [pid for pid, label in zip(subject_ids, labels) if label == 1]

    # Calculate number of not survived subjects for test set based on desired percentage
    test_size = 1 - train_size
    total_test_subjects = int(len(subject_ids) * test_size)
    test_not_survived = int(total_test_subjects * not_survived_pct)
    test_survived = total_test_subjects - test_not_survived

    # Ensure enough subjects in each class
    if test_not_survived > len(not_survived_ids) or test_survived > len(survived_ids):
        raise ValueError(f"Not enough subjects to achieve desired not_survived_pct ({not_survived_pct:.3f}) in test set.")

    # Randomly sample subjects for test set
    random.seed(random_state)
    test_not_survived_ids = random.sample(not_survived_ids, test_not_survived)
    test_survived_ids = random.sample(survived_ids, test_survived)
    test_ids = test_not_survived_ids + test_survived_ids
    train_ids = [pid for pid in subject_ids if pid not in test_ids]

    # Create train and test dictionaries
    train_data = {pid: subjects_data[pid] for pid in train_ids}
    test_data = {pid: subjects_data[pid] for pid in test_ids}

    return train_data, test_data


# New function to evaluate model and plot ROC curve
def evaluate_model(model, test_dataloader, device):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch, pos_enc, labels, masks in test_dataloader:
            batch, pos_enc, labels, masks = batch.to(device), pos_enc.to(device), torch.tensor(labels, dtype=torch.float32).to(device), masks.to(device)

            print(f"DEBUG: masks.shape={masks.shape}")
            p_masks = (masks == 0)
            print(f"DEBUG: p_masks.shape={p_masks.shape}")

            output = model(batch, pos_enc, mask=(masks == 0), mode='classify')  # Mask padded regions
            probs = torch.sigmoid(output.squeeze(1)).cpu().numpy()

            print(f"EVALUATE batch.shape {batch.shape}")
            # for prob, label in zip(probs, labels):
            #     print(f"  prob={prob} label={label}")
            """
            EVALUATE batch.shape torch.Size([5, 1000, 8])
            prob=0.4739331901073456 label=0.0
            prob=0.4690852165222168 label=0.0
            prob=0.47706782817840576 label=1.0
            prob=0.4684579372406006 label=1.0
            prob=0.48032814264297485 label=1.0
            """

            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    # Compute ROC curve and AUROC
    fpr, tpr, _ = roc_curve(all_labels, all_probs, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # --- Find best threshold for F1 ---
    best_thresh, best_f1 = 0, 0
    for thresh in np.linspace(0, 1, 101):  # scan 0..1
        y_pred = (all_probs >= thresh).astype(int)
        f1 = f1_score(all_labels, y_pred)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh

    print(f"Best AUROC threshold for F1 = {best_f1:.3f} at threshold {best_thresh:.2f}")

    num_died = 0
    num_survived = 0
    for label, prob in zip(all_labels, all_probs):
        if label == 1: num_died += 1
        else: num_survived += 1
    print(f"Test actual positive {num_died}")
    print(f"Test actual negative {num_survived}")
    positive_ratio = num_died / (num_died+num_survived)
    print(f"Test proportion positive {positive_ratio:.3f} died")
    print(f"Test proportion positive {1.0-positive_ratio:.3f} survived")



    # # Plot ROC curve
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUROC = {roc_auc:.4f})')
    # plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate (FPR)')
    # plt.ylabel('True Positive Rate (TPR)')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.legend(loc='lower right')
    # plt.grid(True)
    # plt.show()
    #
    # print(f"AUROC: {roc_auc:.4f}")

    charts.plot_auroc(all_labels, all_probs)
    charts.plot_auprc(all_labels, all_probs)

    return roc_auc


def extract_embeddings(model, dataloader, device):
    """
    Get the embeddings for the input dataloader instances.  Useful for visualising the embeddings.
    :param model:
    :param dataloader:
    :param device:
    :return:
    """
    model.eval()  # Set to evaluation mode
    model.to(device)
    embeddings = []
    labels_list = []

    with torch.no_grad():
        # This loop results in DEBUG: masks.shape=torch.Size([10]), which causes error
        # # This loops results in DEBUG: masks.shape=torch.Size([10, 1000]), which is correct
        # for src, pos_enc, labels, masks in dataloader:
        #     src, pos_enc, labels, masks = src.to(device), pos_enc.to(device), torch.tensor(labels, dtype=torch.float32).to(device), masks.to(device)

        for batch in dataloader:
            src, pos_enc, labels, masks = [item.to(device) for item in batch]  # Move to device
            # labels are what?
            # Odd that the code converted labels to torch.Tensor, they are already torch.Tensor???
            # print(f"labels type={type(labels)}") # labels type=<class 'torch.Tensor'>

            # Note global mask only applicable for train/test classifier, not for embeddings?
            print(f"DEBUG: masks.shape={masks.shape}")
            p_masks = (masks == 0)
            print(f"DEBUG: p_masks.shape={p_masks.shape}")
            pooled_emb = model(src, pos_enc, mask=(masks == 0), mode='embed')
            embeddings.append(pooled_emb.cpu().numpy())  # Collect on CPU as NumPy
            labels_list.append(labels.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)  # Shape: (num_samples, num_features)
    labels = np.concatenate(labels_list, axis=0)  # Shape: (num_samples,)
    return embeddings, labels


def visualise_embeddings(embeddings, labels):
    # Reduce to 2D (perplexity=30 is a common default; adjust based on dataset size)
    perplexity = 30
    if len(labels) < 30:
        # perplexity has to be greater than number of samples
        perplexity = int( 0.5*len(labels) )
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate='auto', init='pca', random_state=42)
    embed_2d = tsne.fit_transform(embeddings)  # Shape: (num_samples, 2)

    # Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embed_2d[:, 0], embed_2d[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
    plt.colorbar(scatter, ticks=[0, 1], label='Class Label')
    plt.title('t-SNE Visualization of Time Series Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig('embeddings.png', dpi=300)
    # plt.show()



def print_model_stats(model):
    # Calculate total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    # Calculate total number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {trainable_params}")

def get_device():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # BUG: "mps" no issues for autoencoding but gets nan's for classification training
    #      "cpu" no issues for autoencoding and no issues for classification training
    # return device
    return "cpu"





# Modified main function
def main():

    #---------------------------------------------------------------#
    # Sample data for multiple subjects (replace with your actual data)
    # These parameters are only for generating data
    # They are not molde hyper-parameters
    # ---------------------------------------------------------------#
    num_subjects = 300 # 300 # 1000
    feature_columns = ['glucose', 'potassium', 'sodium', 'creatinine']
    proportion_died = 0.20
    subjects_data, seq_length = dnn_mortality_data.generate_dataset(num_subjects, feature_columns, proportion_died)

    # To stratify training and test percentages of died/survived, compute ratio
    # Used by train-text split
    num_died = 0
    num_survived = 0
    for subject in subjects_data.values():
        if subject['label'] == 1:  num_died += 1
        else: num_survived += 1
    not_survived_pct = num_died / num_survived
    print(f"num_died     = {num_died}")
    print(f"num_survived = {num_survived}")
    print(f"Ratio of died     = {not_survived_pct:.3f}")
    print(f"Ratio of survived = {1.0 - not_survived_pct:.3f}")

    # ---------------------------------------------------------------#
    # Determine global full_length (max time range across all subjects)
    # This is the (max_chart_time - min_chart_time) across all subjects.
    # ---------------------------------------------------------------#
    max_length = 0
    for subject in subjects_data.values():
        df = align_time_series(subject['timeseries'])
        max_length = max(max_length, len(df))
    global_length = max_length



    # ---------------------------------------------------------------#
    # Prepare the dataset for the encoding / feature learning phase
    # ---------------------------------------------------------------#

    # Split data into train and test sets
    # NB: the test_data is aligned/normalised in subjectDataset
    #     but not sequenced (this is only necessary for training the autoencoder)
    train_data, test_data = create_train_test_split(
        subjects_data, train_size=2/3, not_survived_pct=not_survived_pct)

    # Preprocess for autoencoding using training data (align, normalise, sequence)

    auto_dataloader, scaler, num_features = preprocess_for_autoencode(train_data, seq_length)

    # # Demo: Print batch shapes
    # for batch, pos_enc in auto_dataloader:
    #     print(f"Autoencode Batch shape: {batch.shape}")
    #     print(f"Positional encoding shape: {pos_enc.shape}")
    #     break

    device = get_device()

    # Train autoencoder
    print(f"num_features = {num_features} (aka number of features/columns in data")
    model = train_autoencoder(auto_dataloader, num_features, device, epochs=10)


    # ---------------------------------------------------------------#
    # Prepare for the task specific / classification phase using the learned features
    # ---------------------------------------------------------------#
    mask_columns = [f'{col}_mask' for col in feature_columns]
    train_dataset = SubjectDataset(train_data, scaler, global_length, feature_columns, mask_columns, num_features)
    test_dataset = SubjectDataset(test_data, scaler, global_length, feature_columns, mask_columns, num_features)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Visualise the test embeddings (they have labels)
    embeddings, labels = extract_embeddings(model, test_dataloader, device)
    visualise_embeddings(embeddings, labels)

    # classify = False
    # if not classify:
    #     return

    # Train classifier with pos_weight for imbalance (adjust based on train_data)
    num_survived = sum(1 for p in train_data.values() if p['label'] == 0)
    num_died     = sum(1 for p in train_data.values() if p['label'] == 1)
    pos_weight = num_survived / num_died
    train_classifier(train_dataloader, model, device, epochs=10, pos_weight=pos_weight)

    # ---------------------------------------------------------------#
    # Evaluate the classifier on test set, indirectly also evaluating the learned features
    # ---------------------------------------------------------------#
    evaluate_model(model, test_dataloader, device)


if __name__ == "__main__":
    main()
