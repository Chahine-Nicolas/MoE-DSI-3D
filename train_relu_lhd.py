import numpy as np
import torch
import os
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

list_seq = ["A0", "B0", "C0", "D0"] 
root_path = "/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/lidarhd_v2"


class multiSequenceDataset(Dataset):
    def __init__(self, list_seq, data, root_path):
        self.samples = []
        self.labels = []
        self.seq_to_idx = {seq: i for i, seq in enumerate(list_seq)}  # converti 0,2,5 en 0,1,2..

        for seq in list_seq:
            seq_str = seq
            sequence_path = os.path.join(root_path, "256_desc_2025-06-23_11-22-13_run_0_4")
        
            for file_path_i in data[seq_str]:

                file = os.path.basename(file_path_i)[:-4] +'.pt'
                file_path = os.path.join(sequence_path, file)
                
                if os.path.exists(file_path):
                    vec = torch.load(file_path).to(torch.float32) 
                    
                    self.samples.append(vec)
                    self.labels.append(self.seq_to_idx[seq])
        
                    print(file_path_i, self.seq_to_idx[seq])
                               

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


def load_training_set_ids(train_indices_path, val_indices_path):
    print("train_indices_path ", train_indices_path)
    print("val_indices_path ", val_indices_path)
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


"""
class ExpertClassifier(nn.Module):
    def __init__(self, input_dim=256, num_experts=len(list_seq)):
        super(ExpertClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout 30%

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts)
        )

    def forward(self, x):
        return self.model(x)


class ExpertClassifier(nn.Module):
    def __init__(self, input_dim=256, num_experts=len(list_seq)):
        super(ExpertClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, num_experts)
        )
    
    def forward(self, x):
        return self.model(x)
"""


class ExpertClassifier(nn.Module):
    def __init__(self, input_dim=256, num_experts=len(list_seq)):
        super(ExpertClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, num_experts),
        )

    def forward(self, x):
        return self.model(x)



"""
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
"""








        

        

def predict_expert(model, feature_vector, device):
    with torch.no_grad():
        feature_vector = feature_vector.to(device).unsqueeze(0)  # Add batch dimension
        output = model(feature_vector)
        predicted_expert_idx = torch.argmax(output).item()
        print("output", output)
        # probac
        m = nn.Softmax(dim=1)
        prob_seq = m(output)

    return list_seq[predicted_expert_idx], output[0][predicted_expert_idx], prob_seq[0][predicted_expert_idx] 



def main():

    with open("/lustre/fswork/projects/rech/dki/ujo91el/code/these_place_reco/LoGG3D-Net/config/kitti_tuples/is_revisit_D-3_T-30.json") as f:
        data = json.load(f)

    device = 'cuda'
    print("Load dataset")


    """
    train_indicesa = load_training_set_ids("zone_A_dsi_train_list.json", "zone_A_dsi_val_list.json")
    train_indicesb = load_training_set_ids("zone_B_dsi_train_list.json", "zone_B_dsi_val_list.json")
    train_indicesc = load_training_set_ids("zone_C_dsi_train_list.json", "zone_C_dsi_val_list.json")
    train_indicesd = load_training_set_ids("zone_D_dsi_train_list.json", "zone_D_dsi_val_list.json")
    """

    train_indicesa = load_training_set_ids("small_list_A.json", "small_list_A.json")
    train_indicesb = load_training_set_ids("small_list_B.json", "small_list_B.json")
    train_indicesc = load_training_set_ids("small_list_C.json", "small_list_C.json")
    train_indicesd = load_training_set_ids("small_list_D.json", "small_list_D.json")
    
    data = {"A0": train_indicesa, "B0": train_indicesb, "C0": train_indicesc, "D0": train_indicesd}
    
    dataset =  multiSequenceDataset(list_seq, data, root_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

    print("Initialize model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExpertClassifier().to(device)
    
    print("model", model)
    
    print("Define loss and optimizer")
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=0.001)

    optimizer = optim.Adam(model.parameters(), lr=0.002)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

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
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
         = criterion(outputs, labels)
        
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {100 * correct / total:.2f}%")
        
        # Save trained model
        torch.save(model.state_dict(), "expert_router_22.pth")
        
        from collections import Counter
        print("Counter(dataset.labels) ", Counter(dataset.labels))
    
    import pdb; pdb.set_trace()
    model = ExpertClassifier(input_dim=256, num_experts=4).to(device)
    model.load_state_dict(torch.load("expert_router_22.pth", map_location=device))
    model.eval()

    
    # evaluation
    print("start evaluation")
    seq_to_idx = {seq: i for i, seq in enumerate(list_seq)}
    hit, num = 0, 0
    seen_proba = []
    for seq in list_seq:
            sequence_path = os.path.join(root_path, "256_desc_2025-06-23_11-22-13_run_0_4")
            seq_str = seq
        
            for file_path_i in data[seq_str]:
                file = os.path.basename(file_path_i)[:-4] +'.pt'
                file_path = os.path.join(sequence_path, file)
                
                num +=1 
                
                test_feature = torch.load(file_path).to(torch.float32)  # Force float32 
    
                best_expert, score, prob = predict_expert(model, test_feature, device)

                
                print(file_path)
                print(f"Predicted expert: {best_expert}, Expected expert: {seq_str }, Score: {score}, proba: {prob} ")
                seen_proba.append(prob.cpu().numpy())
                
                if best_expert ==  seq_str:
                    hit += 1
                    
    print("correct prediction (%): ", hit / num)
    print("average proba: ", np.mean(seen_proba) )
            

if __name__ == '__main__':
    main()