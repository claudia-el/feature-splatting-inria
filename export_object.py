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
    mask = editing_dict['objects'][0]['affected_gaussian_idx']

    filtered_vertices = vertex[mask]
    new_element = PlyElement.describe(filtered_vertices, 'vertex')

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
    
    
   #extract_object(args.editing_mod, args.ply)







