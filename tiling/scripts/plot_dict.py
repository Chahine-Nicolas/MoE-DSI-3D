import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from PIL import Image

def plot_dict(res_dict, base_name):
    step = 20
    min_x = float('inf')
    max_x = float('-inf')
    min_y = float('inf')
    max_y = float('-inf')

    for value in res_dict.values():
        centroid_query = np.array(value['centroid_query'])
        centroid_res = np.array(value['centroid_res'])

        if centroid_query[0] < min_x:
            min_x = centroid_query[0]
        if centroid_query[0] > max_x:
            max_x = centroid_query[0]
        if centroid_res[0] < min_x:
            min_x = centroid_res[0]
        if centroid_res[0] > max_x:
            max_x = centroid_res[0]

        if centroid_query[1] < min_y:
            min_y = centroid_query[1]
        if centroid_query[1] > max_y:
            max_y = centroid_query[1]
        if centroid_res[1] < min_y:
            min_y = centroid_res[1]
        if centroid_res[1] > max_y:
            max_y = centroid_res[1]


    min_y = 0
    #max_y = 3000
    len_x = max_x - min_x
    len_y = max_y - min_y

    print("len_x:" + str(len_x))
    print("len_y:" + str(len_y))

    
    img_x = math.ceil(len_x / step)
    img_y = math.ceil(len_y / step)

    image_size = (img_x, img_y,3)
    images = {}

    #max_dist=3700
    max_dist=1
    for value in res_dict.values():
        centroid_query = np.array(value['centroid_query'])
        centroid_res = np.array(value['centroid_res'])
        id_eval = value['id_eval']

        distance = np.linalg.norm(centroid_query - centroid_res)

        id_x = math.floor((centroid_query[0] - min_x) / step)
        id_y = math.floor((centroid_query[1] - min_y) / step)

        #print(id_x, id_y, distance)

        if id_eval not in images:
            images[id_eval] = np.zeros(image_size) + 255

        images[id_eval][id_x, id_y,0] = (distance/max_dist)*255
        images[id_eval][id_x, id_y,1] = 0
        images[id_eval][id_x, id_y,2] = 0
        
        if distance > max_dist :
            max_dist = distance

    print(max_dist)
    for id_eval, image in images.items():
        #import pdb; pdb.set_trace()            
        output_path = os.path.join(base_name + '.map_' + str(id_eval) + '.png')
        img = Image.fromarray((image).astype('uint8'))
        new_size = (img.width * 10, img.height * 10)
        resized_img = img.resize(new_size, resample=Image.BILINEAR).rotate(90, expand=True)
        resized_img.save(output_path)
        # print(output_path)
        # plt.imshow(image)
        # plt.savefig(output_path)
        # plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot dictionary function")
    parser.add_argument('--dict_path', required=True, help='Path to the JSON dictionary file')

    args = parser.parse_args()

    with open(args.dict_path, 'r') as fichier:
        res_dict = json.load(fichier)

    base_name = os.path.splitext(os.path.basename(args.dict_path))[0]
    plot_dict(res_dict, args.dict_path)
    print("yolo")
