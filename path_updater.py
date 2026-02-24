import json
import os
import argparse

def clean_dict_values(filepath, file, new_data_path='', save=False):
    print("###################################")
    print("Update path for ", filepath+file)
    with open(filepath+file, "r") as f: 
        data = json.load(f)
    print("original", list(data.items())[0])
    
    cleaner_data = {}
    for key, val in data.items():
        cleaner_data[key] = new_data_path + os.path.basename(val)
    print("clean", list(cleaner_data.items())[0])

    print("Will save as ", filepath+file)
    if save:
        with open(filepath+file, 'w') as f:
            json.dump(cleaner_data, f)
        print("saved at ", filepath+file)
    return

def clean_dict_keys(filepath, file, new_data_path='', save=False):
    print("###################################")
    print("Update path for ", filepath+file)
    with open(filepath+file, "r") as f: 
        data = json.load(f)
    print("original", list(data.items())[0])
    
    cleaner_data = {}
    for key, val in data.items():
        cleaner_data[new_data_path + os.path.basename(key)] = val
    print("clean", list(cleaner_data.items())[0])

    print("Will save as ", filepath+file)
    if save:
        with open(filepath+file, 'w') as f:
            json.dump(cleaner_data, f)
        print("saved at ", filepath+file)
    return

def clean_dict_lists(filepath, file, new_data_path='', save=False):
    print("###################################")
    print("Update path for ", filepath+file)
    with open(filepath+file, "r") as f: 
        data = json.load(f)
    print("original", data[0])
    
    cleaner_list = []
    for i in data:
        cleaner_list.append(new_data_path + os.path.basename(i))

    print("clean", cleaner_list[0])

    print("Will save as ", filepath+file)
    if save:
        with open(filepath+file, 'w') as f:
            json.dump(cleaner_data, f)
        print("saved at ", filepath+file)
    return

def clean_dict_items(filepath, file, new_data_path='', save=False):
    print("###################################")
    print("Update path for ", filepath+file)
    with open(filepath+file, "r") as f: 
        data = json.load(f)
    print("original", list(data.items())[0])
    
    cleaner_data = {}
    for key, val in data.items():
        temp_list = []
        for i in val:
            temp_list.append(new_data_path + os.path.basename(i))
        cleaner_data[new_data_path + os.path.basename(key)] = temp_list
        
    print("clean", list(cleaner_data.items())[0])

    print("Will save as ", filepath+file)
    if save:
        with open(filepath+file, 'w') as f:
            json.dump(cleaner_data, f)
        print("saved at ", filepath+file)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical mapping for KITTI poses")
    
    defaut_dir = "/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/lidarhd_v2/"
    parser.add_argument("--filepath", type=str,  default=defaut_dir, help="Dataset path")
    
    defaut_bin_dir = "/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/lidarhd_v2/bin"
    parser.add_argument("--new_data_path", type=str,  default=defaut_bin_dir, help="Dataset path")
    parser.add_argument("--save", type=bool, default=False, help="Save result as JSON")

    args = parser.parse_args()

    clean_dict_values(args.filepath, "hilbert_12_pad.json", new_data_path=args.new_data_path, save=args.save) #ok
    clean_dict_values(args.filepath, "hilbert_12_pad_val.json",  new_data_path=args.new_data_path, save=args.save) #ok
    clean_dict_keys(args.filepath, "poses_grid2.json", new_data_path=args.new_data_path, save=args.save) #ok 
    clean_dict_lists(args.filepath, "full_list.json", new_data_path=args.new_data_path, save=args.save) #ok 
    clean_dict_lists(args.filepath, "dsi_train_list.json", new_data_path=args.new_data_path, save=args.save) #ok 
    clean_dict_items(args.filepath, "dsi_revisits_lhd_itself.json", new_data_path=args.new_data_path, save=args.save) #ok 
    clean_dict_items(args.filepath, "dsi_revisits_lhd_59mt.json", new_data_path=args.new_data_path, save=args.save) #ok 
