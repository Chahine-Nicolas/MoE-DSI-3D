import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Paramètres de la subdivision
division_size = 32

# Regex pour extraire les informations du nom de fichier
pattern = re.compile(r"LHD_FXX_(\d{4})_(\d{4})_PTS_O_LAMB93_IGN69\.copc_(\d+)_(\d+)_(\d+)\.laz")

def process_lidar_tiles(input_dir):
    # Dictionnaire pour stocker le nombre d'IDs par division
    data = {}

    # Parcourir les fichiers du répertoire spécifié
    for filename in os.listdir(input_dir):
        match = pattern.match(filename)
        if match:
            lat_tile, lon_tile, sub_lat, sub_lon, acquisition_id = map(int, match.groups())
            print(f"lat_tile: {lat_tile}, lon_tile: {lon_tile}, sub_lat: {sub_lat}, sub_lon: {sub_lon}, acquisition_id: {acquisition_id}")
            
            # Position globale de la tuile
            global_lat = lat_tile * division_size + sub_lat
            global_lon = lon_tile * division_size + sub_lon
            
            if (global_lat, global_lon) not in data:
                data[(global_lat, global_lon)] = set()
            
            data[(global_lat, global_lon)].add(acquisition_id)

    # Convertir les données en une matrice
    latitudes = [lat for lat, _ in data.keys()]
    longitudes = [lon for _, lon in data.keys()]

    min_lat, max_lat = min(latitudes), max(latitudes)
    min_lon, max_lon = min(longitudes), max(longitudes)

    matrix = np.zeros((max_lat - min_lat + 1, max_lon - min_lon + 1))

    for (lat, lon), ids in data.items():
        matrix[lon - min_lon,lat - min_lat] = len(ids)
    # Sauvegarde de l'image avec annotations
    fig, ax = plt.subplots(figsize=(40, 40))  # Augmentation de la résolution par 4
    cax = ax.matshow(matrix, cmap='gray', origin='lower')

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            if value > 0:
                ax.text(j, i, str(value), va='center', ha='center', color='red', fontsize=32)  # Augmentation de la taille du texte

    plt.title('Carte des IDs par division de tuile', fontsize=24)
    plt.xlabel('Longitude', fontsize=20)
    plt.ylabel('Latitude', fontsize=20)
    plt.savefig(input_dir + 'lidar_tile_id_map.png', dpi=400)  # Augmentation de la résolution

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génération d'une carte des IDs par division de tuile")
    parser.add_argument("--input_dir", type=str, required=True, help="Répertoire contenant les fichiers LiDAR")
    args = parser.parse_args()
    
    process_lidar_tiles(args.input_dir)
