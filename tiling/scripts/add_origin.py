import argparse
import plyfile
import numpy as np

def process_ply(input_ply, output_ply):
    # Load PLY file
    plydata = plyfile.PlyData.read(input_ply)
    
    # Get the vertex data
    vertices = plydata['vertex']
    
    # Extract x, y, z coordinates and normals
    x = np.array(vertices['x'])
    y = np.array(vertices['y'])
    z = np.array(vertices['z'])
    nx = np.array(vertices['nx'])
    ny = np.array(vertices['ny'])
    nz = np.array(vertices['nz'])

    # Create new coordinates by displacing points along the normal
    x_new = x + 10 * nx
    y_new = y + 10 * ny
    z_new = z + 10 * nz

    # Create new vertex array including original and displaced points
    new_vertices = np.zeros(len(vertices), dtype=[('x_origin', 'f4'), ('y_origin', 'f4'), ('z_origin', 'f4'),
                                                  ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                                  ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])

    new_vertices['x_origin'] = x_new
    new_vertices['y_origin'] = y_new
    new_vertices['z_origin'] = z_new
    new_vertices['x'] = x
    new_vertices['y'] = y
    new_vertices['z'] = z
    new_vertices['nx'] = nx
    new_vertices['ny'] = ny
    new_vertices['nz'] = nz

    # Create new PLY element
    new_element = plyfile.PlyElement.describe(new_vertices, 'vertex')

    # Write the new PLY file
    plydata_out = plyfile.PlyData([new_element])
    plydata_out.write(output_ply)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Displace PLY points along their normals by 10 meters.")
    parser.add_argument("--input_ply", required=True, help="Path to the input PLY file.")
    parser.add_argument("--output_ply", required=True, help="Path to the output PLY file.")
    
    args = parser.parse_args()
    
    process_ply(args.input_ply, args.output_ply)
