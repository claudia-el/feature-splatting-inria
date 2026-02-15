import argparse
import pickle
import numpy as np
from plyfile import PlyElement, PlyData

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=str)
    parser.add_argument("--editing_mod", type=str)


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
    
    object_ply.write('object.ply')




if __name__ == "__main__":

    args = parse_args()

    extract_object(args.editing_mod, args.ply)







