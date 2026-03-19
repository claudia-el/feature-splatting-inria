import numpy as np
import pickle
from plyfile import PlyData, PlyElement
import argparse
from scipy.spatial.transform import Rotation
import os

# created with Claude assistance 

def rotate_scene(ply_path, editing_mod_path):

    with open(editing_mod_path, "rb") as f:
        editing_dict = pickle.load(f)

    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    xyz = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T

    R = np.asarray(editing_dict["scene"]["ground_R"])
    T = np.asarray(editing_dict["scene"]["ground_T"])

    flip_R = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])

    full_R = flip_R @ R


    # transform positions
    xyz = xyz - T
    xyz = xyz @ full_R.T
    xyz = xyz - np.mean(xyz, axis=0)

    # transform quaternions
    R_scene = Rotation.from_matrix(full_R)
    q = np.vstack([vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]]).T
    # 3DGS stores (w, x, y, z), scipy expects (x, y, z, w)
    R_gaussians = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    R_new = R_scene * R_gaussians
    q_new = R_new.as_quat()[:, [3, 0, 1, 2]]  # back to (w, x, y, z)

    # transform normals
    normals = np.vstack([vertex["nx"], vertex["ny"], vertex["nz"]]).T
    normals_new = normals @ full_R.T

    # write output
    new_vertex = np.empty(len(vertex), dtype=vertex.data.dtype)
    new_vertex[:] = vertex.data

    new_vertex["x"] = xyz[:, 0]
    new_vertex["y"] = xyz[:, 1]
    new_vertex["z"] = xyz[:, 2]

    new_vertex["rot_0"] = q_new[:, 0]
    new_vertex["rot_1"] = q_new[:, 1]
    new_vertex["rot_2"] = q_new[:, 2]
    new_vertex["rot_3"] = q_new[:, 3]

    new_vertex["nx"] = normals_new[:, 0]
    new_vertex["ny"] = normals_new[:, 1]
    new_vertex["nz"] = normals_new[:, 2]

    new_element = PlyElement.describe(new_vertex, "vertex")
    base, ext = os.path.splitext(ply_path)
    out_path = base + "_rotated" + ext
    PlyData([new_element], text=False).write(out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--editing_mod", required=True)
    args = parser.parse_args()
    rotate_scene(args.ply, args.editing_mod)