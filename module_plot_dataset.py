# plot current poses
import json
import matplotlib.pyplot as plt
import glob
import os

def plot_dataset(path = '/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/lidarhd_v2/bin/', dataset="full_eval_list.json", poses = "poses_grid2.json", title ='Dataset division', color='green'):  
    f = open(path[:-4] + dataset) 
    dataset_list = json.load(f)
    f.close()
    
    f = open(path[:-4] + poses) 
    data = json.load(f)
    f.close()
    
    x_filter = [data[f][0] for f in dataset_list if f in data]
    y_filter = [data[f][1] for f in dataset_list if f in data]
    
    x_all = [v[0] for v in data.values()]
    y_all = [v[1] for v in data.values()]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(x_filter, y_filter, marker='o', s=5, color=color, label=dataset[:-5])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    return
