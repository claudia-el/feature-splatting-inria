import numpy as np
import pickle
from plyfile import PlyData, PlyElement
import argparse


def rotate_scene(ply_path, editing_mod_path):

    # Load editing metadata
    with open(editing_mod_path, "rb") as f:
        editing_dict = pickle.load(f)

    # Load PLY
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    # Convert to Nx3 array
    xyz = np.vstack([
        vertex["x"],
        vertex["y"],
        vertex["z"]
    ]).T

    # Scene transform
    R = np.asarray(editing_dict["scene"]["ground_R"])
    T = np.asarray(editing_dict["scene"]["ground_T"])

    # -----------------------------
    # Apply inverse scene transform
    # -----------------------------

    # Remove translation
    xyz = xyz - T

    # Remove rotation
    xyz = xyz @ R.T

    # Dataset vertical axis correction
    # (Very common in splatting / NeRF datasets)
    flip_R = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])

    xyz = xyz @ flip_R.T

    # Center scene (optional but usually useful)
    xyz = xyz - np.mean(xyz, axis=0)


    new_vertex = np.empty(len(vertex), dtype=vertex.data.dtype)
    new_vertex[:] = vertex.data

    new_vertex["x"] = xyz[:, 0]
    new_vertex["y"] = xyz[:, 1]
    new_vertex["z"] = xyz[:, 2]

    new_element = PlyElement.describe(new_vertex, "vertex")
    PlyData([new_element], text=False).write(ply_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--ply", required=True)
    parser.add_argument("--editing_mod", required=True)

    args = parser.parse_args()

    rotate_scene(args.ply, args.editing_mod)