import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import hamming_loss, accuracy_score


# ============================================================
# Configuration
# ============================================================

CONFIGS = {
    "EAST": {
        "list_seq": ["A0", "B0", "C0", "D0"],
        "root_path": r"C:/Users/chahi/Desktop/these/code/temp/data/lidarhd_v2/",
        "saved_descriptor_folder": "descriptor",
        "model_name": r"C:\Users\chahi\Desktop\these\code\temp\data\Gate\gate_EAST_new.pth",
    },
    "WEST": {
        "list_seq": ["A1", "B1", "C1", "D1", "E1"],
        "root_path": r"C:/Users/chahi/Desktop/these/code/temp/data/lidarhd_v3/",
        "saved_descriptor_folder": "descriptor",
        "model_name": r"C:\Users\chahi\Desktop\these\code\temp\data\Gate\gate_WEST.pth",
    },
}

REGION = "EAST"

TRAINING = True

BATCH_SIZE = 256
NUM_EPOCHS = 80
LEARNING_RATE = 0.002
INPUT_DIM = 256


# ============================================================
# Dataset
# ============================================================
# ok
class MultiSequenceDataset(Dataset):
    """
    Dataset pour les séquences dont un fichier peut appartenir
    à plusieurs experts.

    Exemple :
        target_label["file.pt"] = [1, 0, 1, 0]

    signifie que le fichier appartient aux experts 0 et 2.
    """

    def __init__(
        self,
        list_seq,
        data,
        root_path,
        target_label,
        saved_descriptor_folder,
    ):
        self.samples = []
        self.labels = []

        sequence_path = os.path.join(
            root_path,
            saved_descriptor_folder
        )

        for seq in list_seq:
            for file_path_i in data[seq]:

                file = os.path.basename(file_path_i)[:-4] + ".pt"
                file_path = os.path.join(sequence_path, file)

                if not os.path.exists(file_path):
                    continue

                if file not in target_label:
                    continue

                vec = torch.load(file_path).to(torch.float32)

                self.samples.append(vec)
                self.labels.append(target_label[file])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


# ============================================================
# Model
# ============================================================
# ok
class ExpertClassifier(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, num_experts),
        )

    def forward(self, x):
        return self.model(x)


# ============================================================
# Chargement des listes train / val / eval
# ============================================================

def load_set_ids(root_path, filename):
    path = os.path.join(root_path, filename)

    print("Loading:", path)

    with open(path, "r") as f:
        return json.load(f)


def load_all_indices(root_path, list_seq, split):
    """
    Charge automatiquement :
        zone_A_dsi_train_list.json
        zone_B_dsi_train_list.json
        ...

    en fonction de list_seq.
    """

    data = {}

    for seq in list_seq:
        zone = seq[0]

        filename = f"zone_{zone}_dsi_{split}_list.json"
        data[seq] = load_set_ids(root_path, filename)

    return data


# ============================================================
# Construction des labels multi-hot
# ============================================================

def build_path_to_ids(all_data, list_seq):
    """
    For all files, gets all the correct experts.

    Example :
        {
            "file1.pt": [0],
            "file2.pt": [0, 2],
            "file3.pt": [1, 3]
        }
    """

    path_to_ids = {}

    for expert_idx, seq in enumerate(list_seq):

        for path in all_data[seq]:

            base = os.path.basename(path)[:-4] + ".pt"

            
            if base not in path_to_ids:
                path_to_ids[base] = []

            if expert_idx not in path_to_ids[base]:
                path_to_ids[base].append(expert_idx)


    return path_to_ids


def build_target_labels(path_to_ids, num_experts):
    """
    Converts lists of experts into multi-hot vectors.

    [0, 2] -> [1, 0, 1, 0]
    """

    target_label = {}

    for filename, expert_ids in path_to_ids.items():

        multi_hot = torch.zeros(
            num_experts,
            dtype=torch.float32
        )

        for expert_id in expert_ids:
            multi_hot[expert_id] = 1.0

        target_label[filename] = multi_hot

    return target_label


# ============================================================
# Chargement complet des données
# ============================================================
# ok
def prepare_data(config):
    list_seq = config["list_seq"]
    root_path = config["root_path"]

    train_data = load_all_indices(
        root_path,
        list_seq,
        "train"
    )

    val_data = load_all_indices(
        root_path,
        list_seq,
        "val"
    )

    eval_data = load_all_indices(
        root_path,
        list_seq,
        "eval"
    )

    print("Train len:", sum(len(v) for v in train_data.values()))
    print("Val len:", sum(len(v) for v in val_data.values()))
    print("Eval len:", sum(len(v) for v in eval_data.values()))

    ########################################################
    # ground truth frame_id distribution
    ########################################################
        
    # On utilise les trois splits pour déterminer tous les
    # experts auxquels chaque fichier appartient.
    all_data = {
        seq: (
            train_data[seq]
            + val_data[seq]
            + eval_data[seq]
        )
        for seq in list_seq
    }

    # dictionnary of correct experts id per files
    path_to_ids = build_path_to_ids(
        all_data,
        list_seq
    )
      
    print("Number of unique files:", len(path_to_ids))
    
    # dictionnary of multi-hot vector per files
    target_label = build_target_labels(path_to_ids, len(list_seq))
    
    return train_data, val_data, eval_data, target_label


# ============================================================
# Prediction
# ============================================================
# ok
def predict_expert(model, feature_vector, device):
    with torch.no_grad():

        feature_vector = (feature_vector.to(device).unsqueeze(0))

        logits = model(feature_vector)
        probs = torch.sigmoid(logits)

        # Choose the best expert
        top1_idx = torch.argmax(probs, dim=1)

        predicted_mask = torch.zeros_like(probs)
        predicted_mask[0, top1_idx] = 1.0

    return (
        predicted_mask[0],
        logits[0],
        probs[0],
    )


# ============================================================
# Entraînement
# ============================================================

def train_model(model, dataloader, dataloader_val, device):
    print("Define loss and optimizer")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(NUM_EPOCHS):

        # -------------------------
        # Training
        # -------------------------

        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for features, labels in dataloader:

            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track performance
            total_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += ((predicted == labels).all(dim=1)).sum().item()

            total += labels.size(0)

        scheduler.step()

        train_accuracy = 100 * correct / total

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Loss: {total_loss:.4f} | "
            f"Accuracy: {train_accuracy:.2f}%"
        )

        # -------------------------
        # Validation
        # -------------------------

        model.eval()

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():

            for features, labels in dataloader_val:

                features = features.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                # Track performance
                total_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct += ((predicted == labels).all(dim=1)).sum().item()
                total += labels.size(0)

        val_accuracy = 100 * correct / total

        print(
            f"Validation | "
            f"Loss: {total_loss:.4f} | "
            f"Accuracy: {val_accuracy:.2f}%"
        )


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, data_eval, target_label, config, device,):
    model.eval()

    list_seq = config["list_seq"]
    root_path = config["root_path"]
    saved_descriptor_folder = config["saved_descriptor_folder"]

    sequence_path = os.path.join(root_path, saved_descriptor_folder)

    hit, num = 0, 0

    seen_proba = []
    # hamming
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for seq in list_seq:
            for file_path_i in data_eval[seq]:
                file = (os.path.basename(file_path_i)[:-4] + ".pt")
                file_path = os.path.join(sequence_path, file)

                if not os.path.exists(file_path):
                    continue

                if file not in target_label:
                    continue

                num += 1

                test_feature = torch.load(file_path).to(torch.float32)
                pred_mask, logits, probs = predict_expert(model, test_feature, device)
                true_label = target_label[file].to(device)
                pred_indices = (pred_mask > 0).nonzero(as_tuple=True)[0].tolist()
                true_indices = (true_label > 0).nonzero(as_tuple=True)[0].tolist()


                correct = bool(set(pred_indices) & set(true_indices))

                if correct:
                    hit += 1

                seen_proba.append(probs.cpu().numpy())

                # For the Hamming loss / Subset accuracy
                all_preds.append((pred_mask > 0).cpu().numpy())
                all_targets.append((true_label > 0).cpu().numpy())

                print(f"\n{file_path}")
                print(
                    f"Pred mask: {pred_mask.cpu().numpy()}, "
                    f"True: {true_label.cpu().numpy()}, "
                    f"Probs: {probs.cpu().numpy()}"
                )

    if num == 0:
        print("No sample to evaluate.")
        return

    hamming = hamming_loss(all_targets, all_preds)

    subset_acc = accuracy_score(all_targets, all_preds)

    print("\n==============================")
    print("Evaluation")
    print("==============================")
    print(f"Hit:               {hit}")
    print(f"Num:               {num}")
    print(f"Hit accuracy:      {hit / num:.2%}")
    print(f"Average probability: {np.mean(seen_proba):.4f}")
    print(f"Hamming Loss:      {hamming:.4f}")
    print(f"Hamming Accuracy:  {1 - hamming:.2%}")
    print(f"Subset Accuracy:   {subset_acc:.2%}")


# ============================================================
# Main
# ============================================================

def main():

    config = CONFIGS[REGION]

    list_seq = config["list_seq"]
    num_experts = len(list_seq)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("==============================")
    print(f"Region: {REGION}")
    print(f"Experts: {list_seq}")
    print(f"Device: {device}")
    print("==============================")

    # -------------------------
    # Data
    # -------------------------
    
    train_data, val_data, eval_data, target_label = prepare_data(config)

    # Train dataset
    dataset = MultiSequenceDataset(
        list_seq,
        train_data,
        config["root_path"],
        target_label,
        config["saved_descriptor_folder"],
    )
    
    # Validation dataset
    dataset_val = MultiSequenceDataset(
        list_seq,
        val_data,
        config["root_path"],
        target_label,
        config["saved_descriptor_folder"],
    )

    # Torch dataloader
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True)

    print("Train samples:", len(dataset))
    print("Validation samples:", len(dataset_val))

    # -------------------------
    # Gate model
    # -------------------------

    model = ExpertClassifier(input_dim=INPUT_DIM, num_experts=num_experts).to(device)

    print(model)

    # -------------------------
    # Gate training
    # -------------------------

    if TRAINING:

        print("\nStarting training...")

        train_model(model, dataloader, dataloader_val, device)

        torch.save(model.state_dict(), config["model_name"])

        print("Model saved:", config["model_name"])

    # -------------------------
    # Load model
    # -------------------------

    print("\nLoading model...")

    model.load_state_dict(torch.load(config["model_name"], map_location=device))

    # -------------------------
    # Evaluation
    # -------------------------

    print("\nStarting evaluation...")

    evaluate_model(model, eval_data, target_label, config, device)


if __name__ == "__main__":
    main()
