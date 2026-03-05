import argparse
import pickle
import numpy as np
from plyfile import PlyElement, PlyData

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

    xyz = np.vstack([
        vertex["x"],
        vertex["y"],
        vertex["z"]
    ]).T

    mask = np.array(editing_dict['objects'][0]['affected_gaussian_idx'])

    R = np.array(editing_dict["scene"]["ground_R"])
    T = np.array(editing_dict["scene"]["ground_T"])

    # Remove scene translation
    xyz = xyz - T

    # Extract object
    obj_xyz = xyz[mask]

    # Remove scene rotation
    obj_xyz = obj_xyz @ R.T

    flip_R = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])

    obj_xyz = obj_xyz @ flip_R.T

    # Center object
    center = np.mean(obj_xyz, axis=0)
    obj_final = obj_xyz - center

    new_vertex = vertex[mask].copy()

    new_vertex["x"] = obj_final[:, 0]
    new_vertex["y"] = obj_final[:, 1]
    new_vertex["z"] = obj_final[:, 2]

    new_element = PlyElement.describe(new_vertex, 'vertex')
    object_ply = PlyData([new_element], text=False)

    obj_name = editing_dict["objects"][0]["name"].split(",")[0]
    object_ply.write(f"{obj_name}.ply")

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







