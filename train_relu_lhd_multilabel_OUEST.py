import numpy as np
import torch
import os
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

list_seq = ["A1", "B1", "C1", "D1", "E1"] 
root_path = "/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/lidarhd_v3"

class multiSequenceDataset(Dataset):
    def __init__(self, list_seq, data, root_path, target_label):
        self.samples = []
        self.labels = []
        self.seq_to_idx = {seq: i for i, seq in enumerate(list_seq)}

        for seq in list_seq:
            seq_str = seq
            sequence_path = os.path.join(root_path, "256_desc_2025-06-23_11-22-13_run_0_4")

            for file_path_i in data[seq_str]:
                file = os.path.basename(file_path_i)[:-4] + '.pt'
                file_path = os.path.join(sequence_path, file)


                if os.path.exists(file_path):
                    vec = torch.load(file_path).to(torch.float32)

                    
                    #  lookup the correct multi-hot label from your dict
                    if file in target_label:
                        label = target_label[file]
                    """
                    else:
                        # if missing, create a zero tensor
                        label = torch.zeros(len(list_seq))
                    """

                    self.samples.append(vec)
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


def load_set_ids(train_indices_path):
    print("train_indices_path ", train_indices_path)
    f = open(root_path +"/"+ train_indices_path) 
    train_indices = json.load(f)
    f.close()
    return train_indices



class SequenceDataset(Dataset):
    def __init__(self, root_path, length_seq):
        self.samples = []
        self.labels = []
        self.length_seq = length_seq  # Lengths of each sequence
        self.cumulative_lengths = self.compute_cumulative_lengths(length_seq)

        # Load data from sequence folder "22"
        sequence_path = os.path.join(root_path, "22", "logg_desc")

        total_samples = sum(length_seq)
        for j in range(total_samples):
            file_path = os.path.join(sequence_path, f"{j:06d}.pt")

            #if j % 5 == 0:  # Skip every 5th sample
                #continue

            if os.path.exists(file_path):
                vec = torch.load(file_path).to(torch.float32)
                self.samples.append(vec)
                self.labels.append(self.get_label_from_index(j))
                
    def compute_cumulative_lengths(self, length_seq):
        """Computes cumulative sequence lengths for mapping indices to labels."""
        cumulative = [0]  # Start from 0
        for length in length_seq:
            cumulative.append(cumulative[-1] + length)
        return cumulative

    def get_label_from_index(self, index):
        """Finds the corresponding sequence label based on index."""
        for i in range(len(self.cumulative_lengths) - 1):
            if self.cumulative_lengths[i] <= index < self.cumulative_lengths[i + 1]:
                return i  # Label is the index in `list_seq`
        return -1  # Should never happen if data is correct

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]



#GATE 2
class ExpertClassifier(nn.Module):
    def __init__(self, input_dim=256, num_experts=len(list_seq)):
        super(ExpertClassifier, self).__init__()
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

            nn.Linear(256, num_experts)
        )
    
    def forward(self, x):
        return self.model(x)



def predict_expert(model, feature_vector, device, threshold=0.5):
    with torch.no_grad():
        feature_vector = feature_vector.to(device).unsqueeze(0)  # [1, 256]
        logits = model(feature_vector)                           # [1, num_experts]
        probs = torch.sigmoid(logits)                            # convert to [0,1] range

        # Choose all experts above threshold
        top1_idx = torch.argmax(probs, dim=1)                    # [1]
        predicted_mask = torch.zeros_like(probs)
        predicted_mask[0, top1_idx] = 1.0     

        
        #predicted_mask = (probs > threshold).float()

    return predicted_mask[0], logits[0], probs[0]



def main():

    with open("/lustre/fswork/projects/rech/dki/ujo91el/code/these_place_reco/LoGG3D-Net/config/kitti_tuples/is_revisit_D-3_T-30.json") as f:
        data = json.load(f)

    device = 'cuda'
    print("Load dataset")



    train_indicesa = load_set_ids("zone_A_dsi_train_list.json")
    train_indicesb = load_set_ids("zone_B_dsi_train_list.json")
    train_indicesc = load_set_ids("zone_C_dsi_train_list.json")
    train_indicesd = load_set_ids("zone_D_dsi_train_list.json")
    train_indicese = load_set_ids("zone_E_dsi_train_list.json")
    
    data = {"A1": train_indicesa, "B1": train_indicesb, "C1": train_indicesc, "D1": train_indicesd, "E1": train_indicese}


    val_indicesa = load_set_ids("zone_A_dsi_val_list.json")
    val_indicesb = load_set_ids("zone_B_dsi_val_list.json")
    val_indicesc = load_set_ids("zone_C_dsi_val_list.json")
    val_indicesd = load_set_ids("zone_D_dsi_val_list.json")
    val_indicese = load_set_ids("zone_E_dsi_val_list.json")

    data_val = {"A1": val_indicesa, "B1": val_indicesb, "C1": val_indicesc, "D1": val_indicesd, "E1": val_indicese}

    
    eval_indicesa = load_set_ids("zone_A_dsi_eval_list.json")
    eval_indicesb = load_set_ids("zone_B_dsi_eval_list.json")
    eval_indicesc = load_set_ids("zone_C_dsi_eval_list.json")
    eval_indicesd = load_set_ids("zone_D_dsi_eval_list.json")
    eval_indicese = load_set_ids("zone_E_dsi_eval_list.json")

    data_eval = {"A1": eval_indicesa, "B1": eval_indicesb, "C1": eval_indicesc, "D1": eval_indicesd, "E1": eval_indicese}

    
    print("train len ", len(train_indicesa + train_indicesb + train_indicesc + train_indicesd + train_indicese ) )
    print("val len ", len(val_indicesa + val_indicesb + val_indicesc + val_indicesd + val_indicese) )
    print("eval len ", len(eval_indicesa + eval_indicesb + eval_indicesc + eval_indicesd + eval_indicese))
    import pdb; pdb.set_trace()
    ########################################################
    # ground truth frame_id distribution
    ########################################################
    path_to_ids = {}

    all_names = [
        train_indicesa,
        train_indicesb,
        train_indicesc,
        train_indicesd,
        train_indicese
    ]

    
    for idx, name_list in enumerate(all_names):
        for path in name_list:
            base = os.path.basename(path)[:-4] + '.pt'  # extract filename
            if base not in path_to_ids:
                path_to_ids[base] = []
            path_to_ids[base].append(idx)


    
    all_names = [
        val_indicesa,
        val_indicesb,
        val_indicesc,
        val_indicesd,
        val_indicese
    ]

    
    for idx, name_list in enumerate(all_names):
        for path in name_list:
            base = os.path.basename(path)[:-4] + '.pt'   # extract filename
            if base not in path_to_ids:
                path_to_ids[base] = []
            path_to_ids[base].append(idx)


    all_names = [
        eval_indicesa,
        eval_indicesb,
        eval_indicesc,
        eval_indicesd,
        eval_indicese
    ]

    
    for idx, name_list in enumerate(all_names):
        for path in name_list:
            base = os.path.basename(path)[:-4] + '.pt'   # extract filename
            if base not in path_to_ids:
                path_to_ids[base] = []
            path_to_ids[base].append(idx)

    print(len(path_to_ids))
    
    expert_labels = path_to_ids

    
    ########################################################
    
    num_experts = len(list_seq)

    target_label = {}

    for key, val in path_to_ids.items():
        multi_hot = torch.zeros(num_experts, dtype=torch.float32)
        for valid_expert in val:
            multi_hot[valid_expert] = 1.0
        target_label[key] = multi_hot

    print("loading train set")
    dataset =  multiSequenceDataset(list_seq, data, root_path, target_label)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    print("loaded train set")

    print("loading val set")
    dataset_val =  multiSequenceDataset(list_seq, data, root_path, target_label)
    dataloader_val = DataLoader(dataset, batch_size=256, shuffle=True)
    print("loaded val set")
    
    print("loading eval set")
    dataset_eval =  multiSequenceDataset(list_seq, data, root_path, target_label)
    dataloader_eval = DataLoader(dataset, batch_size=256, shuffle=True)
    print("loaded eval set")
    


    print("Initialize model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExpertClassifier(input_dim=256, num_experts=5).to(device)
 
    
    print("Define loss and optimizer")
    criterion = nn.BCEWithLogitsLoss()


    optimizer = optim.Adam(model.parameters(), lr=0.002)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


    model_name = "expert_gate_ouest0.pth"
    
    training=True
    training=False
    if training:
        print("Training loop")
        num_epochs = 80
        for epoch in range(num_epochs):
            total_loss = 0
            correct = 0
            total = 0
        
            for features, labels in dataloader:
                features, labels = features.to(device), labels.to(device)
                
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

            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {100 * correct / total:.2f}%")

            # validation
            total_loss = 0
            correct = 0
            total = 0
            for features, labels in dataloader_val:
                features, labels = features.to(device), labels.to(device)
        
                # Forward pass
                outputs = model(features)
                loss_valid = criterion(outputs, labels)
        
                # Track performance
                total_loss += loss_valid.item()
    
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct += ((predicted == labels).all(dim=1)).sum().item()
                total += labels.size(0)
                
            print(f"Validation, Loss: {total_loss:.4f}, Accuracy: {100 * correct / total:.2f}%")
        
        # Save trained model
        torch.save(model.state_dict(), model_name)
        
        from collections import Counter
        print("Counter(dataset.labels) ", Counter(dataset.labels))
    

    model = ExpertClassifier(input_dim=256, num_experts=5).to(device)
    model.load_state_dict(torch.load(model_name, map_location=device))
    model.eval()




    
    
    # evaluation
    print("start evaluation")
    seq_to_idx = {seq: i for i, seq in enumerate(list_seq)}
    hit, num = 0, 0
    seen_proba = []
    for seq in list_seq:
            sequence_path = os.path.join(root_path, "256_desc_2025-06-23_11-22-13_run_0_4")
            seq_str = seq
        
            for file_path_i in data_eval[seq_str]:
                file = os.path.basename(file_path_i)[:-4] +'.pt'
                file_path = os.path.join(sequence_path, file)
                
                num +=1 
                
                test_feature = torch.load(file_path).to(torch.float32)  # Force float32 
                #best_expert, score, prob = predict_expert(model, test_feature, device)


                pred_mask, logits, probs = predict_expert(model, test_feature, device, threshold=0.5)
                true_label = target_label[file].to(device)
        
                # Evaluate correctness

                pred_indices = (pred_mask > 0).nonzero(as_tuple=True)[0].tolist()
                true_indices = (true_label > 0).nonzero(as_tuple=True)[0].tolist()
                correct =  bool(set(pred_indices) & set(true_indices))
        
                if correct:
                    hit += 1
        
                seen_proba.append(probs.cpu().numpy())
        
                print(f"{file_path}")
                print(f"Pred mask: {pred_mask.cpu().numpy()}, True: {true_label.cpu().numpy()}, "
                      f"Probs: {probs.cpu().numpy()}")
        
                print(f"Exact-match accuracy: {hit / num:.2%}")
                print(f"Average probability: {np.mean(seen_proba):.4f}")


if __name__ == '__main__':
    main()
