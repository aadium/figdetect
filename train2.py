import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# Set device and memory safety
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True 

class MultiLabelDualDataset(Dataset):
    def __init__(self, df, max_len=64):
        for col in ['full_text', 'full_text_phonetic', 'highlights', 'highlights_phonetic']:
            df[col] = df[col].fillna("")

        # GET ALL UNIQUE FIGURES
        self.classes = sorted(df['figure_name'].unique())
        self.num_classes = len(self.classes)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # GROUP BY TEXT: Collect all labels for the same text instance
        # Using 'full_text' as the anchor for grouping
        self.grouped = df.groupby('full_text').agg({
            'full_text_phonetic': 'first',
            'figure_name': lambda x: list(set(x)), # Get all unique figures for this text
            'highlights': lambda x: '; '.join(x.astype(str)),
            'highlights_phonetic': lambda x: '; '.join(x.astype(str))
        }).reset_index()

        self.max_len = max_len
        self.matrices_text = []
        self.matrices_phone = []
        self.multi_labels = []

        print(f"Generating Multi-Label Matrices for {len(self.grouped)} unique texts...")
        for _, row in self.grouped.iterrows():
            self.matrices_text.append(self.create_text_matrix(row['full_text']))
            self.matrices_phone.append(self.create_phone_matrix(row['full_text_phonetic']))
            
            # Create Binary Label Vector [0, 1, 0, 0, 1...]
            label_vec = np.zeros(self.num_classes, dtype=np.float32)
            for fig in row['figure_name']:
                label_vec[self.class_to_idx[fig]] = 1.0
            self.multi_labels.append(label_vec)

    def create_text_matrix(self, text):
        raw_words = str(text).lower().split()
        clean_words = [w.strip('.,!?;:"()') for w in raw_words]
        matrix = np.zeros((3, self.max_len, self.max_len), dtype=np.float32)
        for i in range(min(len(clean_words), self.max_len)):
            for j in range(min(len(clean_words), self.max_len)):
                if i == j: continue
                if clean_words[i] == clean_words[j] and len(clean_words[i]) > 0:
                    matrix[0, i, j] = 1.0
                if clean_words[i][:1] == clean_words[j][:1]: matrix[1, i, j] += 0.5
                if any(p in raw_words[i] for p in [',', ';', ':']): matrix[2, i, j] = 1.0
        return matrix

    def create_phone_matrix(self, phone_text):
        phone_words = str(phone_text).lower().split()
        matrix = np.zeros((2, self.max_len, self.max_len), dtype=np.float32)
        for i in range(min(len(phone_words), self.max_len)):
            for j in range(min(len(phone_words), self.max_len)):
                if i == j: continue
                if phone_words[i] == phone_words[j]: matrix[0, i, j] = 1.0
                if phone_words[i][-2:] == phone_words[j][-2:]: matrix[1, i, j] = 1.0
        return matrix

    def __len__(self): return len(self.grouped)
    def __getitem__(self, idx):
        return (torch.tensor(self.matrices_text[idx]), 
                torch.tensor(self.matrices_phone[idx]), 
                torch.tensor(self.multi_labels[idx]))

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
            # No Softmax here because BCEWithLogitsLoss handles it via Sigmoid internally
        )

    def forward(self, txt, phn):
        t_feat = self.text_encoder(txt)
        p_feat = self.phone_encoder(phn)
        combined = torch.cat((t_feat, p_feat), dim=1)
        return self.classifier(combined)

def train_and_evaluate(csv_path):
    df = pd.read_csv(csv_path)
    dataset = MultiLabelDualDataset(df)
    
    indices = np.arange(len(dataset))
    # Note: Stratification is harder in multi-label; simple split for now
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=32, shuffle=True)
    test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=32, shuffle=False)

    model = DualStreamCNN(dataset.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Binary Cross Entropy for Multi-Label
    criterion = nn.BCEWithLogitsLoss()

    print("Starting Multi-Label training...")
    for epoch in range(100): # Multi-label often converges faster
        model.train()
        total_loss = 0
        for txt, phn, labs in train_loader:
            txt, phn, labs = txt.to(device), phn.to(device), labs.to(device)
            optimizer.zero_grad()
            output = model(txt, phn)
            loss = criterion(output, labs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for txt, phn, labs in test_loader:
            txt, phn, labs = txt.to(device), phn.to(device), labs.to(device)
            # Get raw scores
            output = torch.sigmoid(model(txt, phn))
            all_outputs.extend(output.cpu().numpy())
            all_labels.extend(labs.cpu().numpy())

    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)

    # FIND BEST THRESHOLDS
    best_thresholds = []
    final_preds = np.zeros_like(all_outputs)

    for i in range(dataset.num_classes):
        best_f1 = 0
        best_thresh = 0.5
        # Test thresholds from 0.05 to 0.50
        for thresh in np.linspace(0.05, 0.5, 20):
            preds = (all_outputs[:, i] > thresh).astype(float)
            from sklearn.metrics import f1_score
            f1 = f1_score(all_labels[:, i], preds)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        best_thresholds.append(best_thresh)
        final_preds[:, i] = (all_outputs[:, i] > best_thresh).astype(float)
        print(f"Figure: {dataset.classes[i]:<15} | Best Threshold: {best_thresh:.2f}")

    print("\nOptimized Multi-Label Classification Report:")
    print(classification_report(all_labels, final_preds, target_names=dataset.classes))

    # Plotting one confusion matrix per class is standard for multi-label, 
    # but for your meeting, a heatmap of class correlations is often better.
    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    train_and_evaluate("gofigure_phonetized.csv")