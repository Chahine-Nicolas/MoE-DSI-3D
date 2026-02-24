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



"""

#GATE 1
class ExpertClassifier(nn.Module):
    def __init__(self, input_dim=256, num_experts=len(list_seq)):
        super(ExpertClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, num_experts),
        )

    def forward(self, x):
        return self.model(x)



"""


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
        

"""
class ExpertClassifier(nn.Module):
    def __init__(self, input_dim=256, num_experts=len(list_seq)):
        super(ExpertClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
        
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
        
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
        
            nn.Linear(128, num_experts)
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



    train_indicesa = load_set_ids("zone_A_dsi_train_list.json")[:64]
    train_indicesb = load_set_ids("zone_B_dsi_train_list.json")[:64]
    train_indicesc = load_set_ids("zone_C_dsi_train_list.json")[:64]
    train_indicesd = load_set_ids("zone_D_dsi_train_list.json")[:64] 

    data = {"A0": train_indicesa, "B0": train_indicesb, "C0": train_indicesc, "D0": train_indicesd}
    dataset =  multiSequenceDataset(list_seq, data, root_path)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    
    val_indicesa = load_set_ids("zone_A_dsi_val_list.json")[:64] 
    val_indicesb = load_set_ids("zone_B_dsi_val_list.json")[:64] 
    val_indicesc = load_set_ids("zone_C_dsi_val_list.json")[:64] 
    val_indicesd = load_set_ids("zone_D_dsi_val_list.json")[:64] 

    data_val = {"A0": val_indicesa, "B0": val_indicesb, "C0": val_indicesc, "D0": val_indicesd}
    dataset_val =  multiSequenceDataset(list_seq, data, root_path)
    dataloader_val = DataLoader(dataset, batch_size=256, shuffle=True)

    

    eval_indicesa = load_set_ids("zone_A_dsi_eval_list.json")[:64] 
    eval_indicesb = load_set_ids("zone_B_dsi_eval_list.json")[:64] 
    eval_indicesc = load_set_ids("zone_C_dsi_eval_list.json")[:64] 
    eval_indicesd = load_set_ids("zone_D_dsi_eval_list.json")[:64] 

    data_eval = {"A0": eval_indicesa, "B0": eval_indicesb, "C0": eval_indicesc, "D0": eval_indicesd}
    dataset_eval =  multiSequenceDataset(list_seq, data, root_path)
    dataloader_eval = DataLoader(dataset, batch_size=256, shuffle=True)
   
    data = {"A0": eval_indicesa, "B0": eval_indicesb, "C0": eval_indicesc, "D0": eval_indicesd}
    dataset =  multiSequenceDataset(list_seq, data, root_path)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)


    """
    (Pdb) train_indicesa[0]
    '/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/lidarhd_v2/bin/LHD_FXX_0656_6860_PTS_O_LAMB93_IGN69.copc_10_10_46.bin'
    (Pdb) val_indicesa[0]
    '/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/lidarhd_v2/bin/LHD_FXX_0656_6860_PTS_O_LAMB93_IGN69.copc_10_10_47.bin'
    (Pdb) eval_indicesa[0]
    '/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/lidarhd_v2/bin/LHD_FXX_0656_6860_PTS_O_LAMB93_IGN69.copc_10_10_48.bin'
    """

    

    
    eval_indicesa = load_set_ids("small_list_A.json")[:64] 
    eval_indicesb = load_set_ids("small_list_B.json")[:64] 
    eval_indicesc = load_set_ids("small_list_C.json")[:64] 
    eval_indicesd = load_set_ids("small_list_D.json")[:64] 

    data_eval = {"A0": train_indicesa, "B0": train_indicesb, "C0": train_indicesc, "D0": train_indicesd}
    dataset_eval =  multiSequenceDataset(list_seq, data, root_path)
    dataloader_eval = DataLoader(dataset, batch_size=256, shuffle=True)

    import pdb; pdb.set_trace()

    """
    (Pdb) train_indicesa[0]
    '/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/lidarhd_v2/bin/LHD_FXX_0656_6860_PTS_O_LAMB93_IGN69.copc_10_10_46.bin'
    (Pdb) val_indicesa[0]
    '/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/lidarhd_v2/bin/LHD_FXX_0656_6860_PTS_O_LAMB93_IGN69.copc_10_10_47.bin'
    (Pdb) eval_indicesa[0]
    '/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/lidarhd_v2/bin/LHD_FXX_0656_6860_PTS_O_LAMB93_IGN69.copc_17_10_47.bin'
    (Pdb) 
    """

    
    """
    
    # mini dataset

    train_indicesa = load_training_set_ids("small_list_A.json", "small_list_A.json")
    train_indicesb = load_training_set_ids("small_list_B.json", "small_list_B.json")
    train_indicesc = load_training_set_ids("small_list_C.json", "small_list_C.json")
    train_indicesd = load_training_set_ids("small_list_D.json", "small_list_D.json")
 
    
    data = {"A0": train_indicesa, "B0": train_indicesb, "C0": train_indicesc, "D0": train_indicesd}

    data = {"A0": train_indicesa[:8], "B0": train_indicesb[:8], "C0": train_indicesc[:8], "D0": train_indicesd[:8]}
    data_eval = {"A0": train_indicesa[8:], "B0": train_indicesb[8:], "C0": train_indicesc[8:], "D0": train_indicesd[8:]}
    
    dataset =  multiSequenceDataset(list_seq, data, root_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    dataset_eval =  multiSequenceDataset(list_seq, data, root_path)
    dataloader_eval = DataLoader(dataset_eval, batch_size=32, shuffle=True)
    """


    
    #dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
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


    model_name = "expert_router_4linear_all.pth"
    
    training=True
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
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
            print(f"Validation, Loss: {total_loss:.4f}, Accuracy: {100 * correct / total:.2f}%")
        
        # Save trained model
        torch.save(model.state_dict(), model_name)
        
        from collections import Counter
        print("Counter(dataset.labels) ", Counter(dataset.labels))
    
    import pdb; pdb.set_trace()
    model = ExpertClassifier(input_dim=256, num_experts=4).to(device)
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