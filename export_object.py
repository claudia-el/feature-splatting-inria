import argparse
import pickle
import numpy as np
from plyfile import PlyElement, PlyData
from scipy.spatial.transform import Rotation

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=str)
    parser.add_argument("--editing_mod", type=str)
    parser.add_argument("--room", type=bool)
    return parser.parse_args()


def load_pkl(editing_mod_path):
    with open(editing_mod_path, "rb") as f:
        return pickle.load(f)
    

def extract_object(editing_mod_path, ply_path):
    editing_dict = load_pkl(editing_mod_path)
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    xyz = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T
    mask = np.array(editing_dict['objects'][0]['affected_gaussian_idx'])

    R = np.array(editing_dict["scene"]["ground_R"])
    T = np.array(editing_dict["scene"]["ground_T"])

    flip_R = np.array([
            [1, 0,  0],
            [0, -1, 0],
            [0, 0, -1]
        ])

    # Remove translation
    xyz = xyz - T
    # Filter points
    obj_xyz = xyz[mask]
    # p′=pR^T (row vector rotation rules)
    obj_xyz = obj_xyz @ R.T      
    obj_xyz = obj_xyz @ flip_R.T 

   
    center = np.mean(obj_xyz, axis=0)
    obj_final = obj_xyz - center

    combined_R = flip_R @ R
    # converts 3×3 rotation matrix into a SciPy rotation object
    R_scene = Rotation.from_matrix(combined_R)

    # Reshape data into N × 4 array
    q = np.vstack([
        vertex["rot_0"][mask],
        vertex["rot_1"][mask],
        vertex["rot_2"][mask],
        vertex["rot_3"][mask]
    ]).T
    
    # wxyz -> xyzw  (expected by SciPy)
    R_gaussians = Rotation.from_quat(q[:, [1, 2, 3, 0]])  
    # apply local (relative to object) and scene rotatation to guassians
    R_new = R_scene * R_gaussians
    # xyzw -> wxyz
    q_new = R_new.as_quat()[:, [3, 0, 1, 2]]              

    normals = np.vstack([
        vertex["nx"][mask],
        vertex["ny"][mask],
        vertex["nz"][mask]
    ]).T
    normals_new = (normals @ combined_R.T)

    new_vertex = vertex[mask].copy()

    new_vertex["x"] = obj_final[:, 0]
    new_vertex["y"] = obj_final[:, 1]
    new_vertex["z"] = obj_final[:, 2]

    new_vertex["rot_0"] = q_new[:, 0]
    new_vertex["rot_1"] = q_new[:, 1]
    new_vertex["rot_2"] = q_new[:, 2]
    new_vertex["rot_3"] = q_new[:, 3]

    new_vertex["nx"] = normals_new[:, 0]
    new_vertex["ny"] = normals_new[:, 1]
    new_vertex["nz"] = normals_new[:, 2]

    new_element = PlyElement.describe(new_vertex, 'vertex')
    PlyData([new_element], text=False).write(
        f"{editing_dict['objects'][0]['name'].split(',')[0]}.ply"
    )

def room(editing_mod_path, ply_path):
    editing_dict = load_pkl(editing_mod_path)
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    
    mask = editing_dict['objects'][0]['affected_gaussian_idx']
    room_vertices = vertex[~mask]

    new_element = PlyElement.describe(room_vertices, 'vertex')
    room_ply = PlyData([new_element], text=False)

    obj_name = editing_dict["objects"][0]["name"].split(",")[0]
    room_ply.write(f"room_without_{obj_name}.ply")


if __name__ == "__main__":
    args = parse_args()

    if args.room:
        room(args.editing_mod, args.ply)
    
    extract_object(args.editing_mod, args.ply)