import os
import numpy as np
import argparse
import open3d as o3d
import laspy

class PointCloudProcessor:
    def __init__(self):
        pass

    def save_pointcloud_tensor(self, output_dir, filename, tensor):
        """Sauvegarde un tensor sous forme binaire en conservant le nom du fichier d'origine."""
        base_name = os.path.splitext(filename)[0]  # Récupérer le nom sans extension
        bin_filename = os.path.join(output_dir, f"{base_name}.bin")
        tensor.astype(np.float32).tofile(bin_filename)

    def laz_to_tensor(self, laz_file):
        """Convertit un fichier .laz en un tensor numpy de type float32."""
        with laspy.open(laz_file) as las:
            points = las.read()
            xyz = np.vstack((points.x, points.y, points.z)).T  # Extraire (x, y, z)

            # Vérifier si l'intensité est présente
            if hasattr(points, 'intensity'):
                intensity = points.intensity[:, np.newaxis]  # Ajouter une dimension pour l'intensité
            else:
                intensity = np.zeros((xyz.shape[0], 1))  # Valeurs nulles si pas d'intensité

            return np.hstack((xyz, intensity))  # Concaténer en une seule matrice


def process_directory(input_dir, output_dir):
    """Parcourt le répertoire d'entrée et convertit chaque fichier .laz en .bin."""
    os.makedirs(output_dir, exist_ok=True)
    
    processor = PointCloudProcessor()
    laz_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".laz")])  # Trier pour un ordre stable

    for filename in laz_files:
        laz_file = os.path.join(input_dir, filename)

        # Convertir le fichier LAZ en tensor
        tensor = processor.laz_to_tensor(laz_file)

        # Sauvegarder sous forme binaire avec le même nom
        processor.save_pointcloud_tensor(output_dir, filename, tensor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LAZ files to Velodyne binary format.")
    parser.add_argument("--input_dir", type=str, help="Directory containing the LAZ files.")
    parser.add_argument("--output_dir", type=str, help="Directory to save the converted BIN files.")
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir)
