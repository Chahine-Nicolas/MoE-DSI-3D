import os
import numpy as np
import argparse
import open3d as o3d
import laspy

class PointCloudProcessor:
    def __init__(self):
        pass

    def save_pointcloud_ply(self, output_dir, filename, points):
        """Sauvegarde un nuage de points au format PLY."""
        base_name = os.path.splitext(filename)[0]  # Récupérer le nom sans extension
        ply_filename = os.path.join(output_dir, f"{base_name}.ply")
        
        # Créer un nuage de points Open3D et sauvegarder en PLY
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # Coordonnées (x, y, z)
        
        if points.shape[1] > 3:
            intensities = points[:, 3]  # Si l'intensité est présente
            pcd.colors = o3d.utility.Vector3dVector(np.tile(intensities[:, None], (1, 3)) / 65535.0)  # Normaliser l'intensité sur [0, 1]

        o3d.io.write_point_cloud(ply_filename, pcd)

    def laz_to_points(self, laz_file):
        """Convertit un fichier .laz en un tableau numpy de points (x, y, z, intensity)."""
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
    """Parcourt le répertoire d'entrée et convertit chaque fichier .laz en .ply."""
    os.makedirs(output_dir, exist_ok=True)

    processor = PointCloudProcessor()
    laz_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".laz")])  # Trier pour un ordre stable

    for filename in laz_files:
        laz_file = os.path.join(input_dir, filename)

        # Convertir le fichier LAZ en tableau de points
        points = processor.laz_to_points(laz_file)

        # Sauvegarder le nuage de points au format PLY
        processor.save_pointcloud_ply(output_dir, filename, points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LAZ files to PLY format.")
    parser.add_argument("--input_dir", type=str, help="Directory containing the LAZ files.")
    parser.add_argument("--output_dir", type=str, help="Directory to save the converted PLY files.")
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir)
