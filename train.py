import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# Set device and memory safety
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True 

class DualStreamDataset(Dataset):
    def __init__(self, df, max_len=64):
        # Fill NaNs to prevent processing errors
        for col in ['full_text', 'full_text_phonetic', 'highlights', 'highlights_phonetic']:
            df[col] = df[col].fillna("")

        # Group by annotation_id to handle multi-part highlights
        self.grouped = df.groupby('annotation_id').agg({
            'full_text': 'first',
            'full_text_phonetic': 'first',
            'figure_name': 'first',
            'highlights': lambda x: '; '.join(x.astype(str)),
            'highlights_phonetic': lambda x: '; '.join(x.astype(str))
        }).reset_index()

        self.encoder = LabelEncoder()
        self.labels = self.encoder.fit_transform(self.grouped['figure_name'])
        self.max_len = max_len
        self.matrices_text = []
        self.matrices_phone = []

        print(f"Generating Dual Matrices for {len(self.grouped)} samples...")
        for _, row in self.grouped.iterrows():
            self.matrices_text.append(self.create_text_matrix(row['full_text'], row['highlights']))
            self.matrices_phone.append(self.create_phone_matrix(row['full_text_phonetic'], row['highlights_phonetic']))

    def create_text_matrix(self, text, highlights):
        raw_words = str(text).lower().split()
        clean_words = [w.strip('.,!?;:"()') for w in raw_words]
        matrix = np.zeros((3, self.max_len, self.max_len), dtype=np.float32)

        for i in range(min(len(clean_words), self.max_len)):
            for j in range(min(len(clean_words), self.max_len)):
                if i == j: continue
                # CH 0: Word Identity
                if clean_words[i] == clean_words[j] and len(clean_words[i]) > 0:
                    matrix[0, i, j] = 1.0
                # CH 1: Visual Alliteration (Starts)
                if clean_words[i][:1] == clean_words[j][:1]: matrix[1, i, j] += 0.5
                # CH 2: Punctuation Boundaries
                if any(p in raw_words[i] for p in [',', ';', ':']): matrix[2, i, j] = 1.0
        return matrix

    def create_phone_matrix(self, phone_text, phone_highlights):
        phone_words = str(phone_text).lower().split()
        matrix = np.zeros((2, self.max_len, self.max_len), dtype=np.float32)

        for i in range(min(len(phone_words), self.max_len)):
            for j in range(min(len(phone_words), self.max_len)):
                if i == j: continue
                # CH 0: Phonetic Identity (Repetition of exact sounds)
                if phone_words[i] == phone_words[j]:
                    matrix[0, i, j] = 1.0
                # CH 1: Phonetic Suffix/Coda (Rhyme/Assonance detector)
                if phone_words[i][-2:] == phone_words[j][-2:]:
                    matrix[1, i, j] = 1.0
        return matrix

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return (torch.tensor(self.matrices_text[idx]), 
                torch.tensor(self.matrices_phone[idx]), 
                torch.tensor(self.labels[idx]))

class StreamEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
    def forward(self, x): return self.conv(x)

class DualStreamCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.text_encoder = StreamEncoder(3)
        self.phone_encoder = StreamEncoder(2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((64 * 4 * 4) * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, txt, phn):
        t_feat = self.text_encoder(txt)
        p_feat = self.phone_encoder(phn)
        combined = torch.cat((t_feat, p_feat), dim=1)
        return self.classifier(combined)

def train_and_evaluate(csv_path):
    df = pd.read_csv(csv_path)
    dataset = DualStreamDataset(df)
    
    indices = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=dataset.labels, random_state=42)
    
    # Balancing the classes
    class_counts = np.bincount(dataset.labels[train_idx])
    weights = 1. / class_counts
    samples_weights = torch.from_numpy(weights[dataset.labels[train_idx]])
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))
    
    train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
    # Subset test loader to maintain original distribution
    test_subset = torch.utils.data.Subset(dataset, test_idx)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

    model = DualStreamCNN(len(dataset.encoder.classes_)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print("Starting training loop...")
    try:
        for epoch in range(400):
            model.train()
            total_loss = 0
            for txt, phn, labs in train_loader:
                txt, phn, labs = txt.to(device), phn.to(device), labs.to(device)
                optimizer.zero_grad()
                loss = criterion(model(txt, phn), labs)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch+1) % 5 == 0:
                print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")
    except KeyboardInterrupt:
        print("Training interrupted. Generating metrics on current state...")

    # Evaluation & Confusion Matrix
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for txt, phn, labs in test_loader:
            txt, phn, labs = txt.to(device), phn.to(device), labs.to(device)
            output = model(txt, phn)
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labs.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=dataset.encoder.classes_))

    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=dataset.encoder.classes_, 
                yticklabels=dataset.encoder.classes_)
    plt.title('Dual-Stream Rhetorical Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('dual_cnn_confusion_matrix.png')
    plt.show()

    # Final Cleanup for the meeting
    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    train_and_evaluate("gofigure_phonetized.csv")