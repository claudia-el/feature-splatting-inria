import os
import torch
import argparse
import numpy as np
import open3d as o3d
from arguments import ModelParams, get_combined_args
from segment import select_gs_for_phys
from scene import Scene
from gaussian_renderer import GaussianModel



def save_splat(xyz, features, out_path):
    
    torch.save({'xyz': xyz, 'features': features}, out_path)

def save_ply(xyz, out_path):
    
    #intializes point cloud object
    pc_obj = o3d.geometry.PointCloud()
    #wraps NumPy array into an Open3D-compatible vector 
    pc_obj.points = o3d.utility.Vector3dVector(xyz.cpu().numpy()) #saves as NumPy array
    #Writes to output path
    o3d.io.write_point_cloud(out_path, pc_obj)

def extract_objects(model_path, iteration, fg_obj_list, bg_obj_list, ground_plane_name,
                    rigid_object_name, threshold, object_select_eps, inward_bbox_offset,
                    final_noise_filtering, interactive_viz, output, save_ply_flag=True):
    
    os.makedirs(output, exist_ok=True)

    #command line argument parser 
    parser = argparse.ArgumentParser()

    model = ModelParams(parser, sentinel=True)
    args = get_combined_args(parser)
    args.model_path = model_path
    args.iteration = iteration

    fg_obj_list = fg_obj_list if isinstance(fg_obj_list, list) else fg_obj_list.split(",")
    bg_obj_list = bg_obj_list if isinstance(bg_obj_list, list) else bg_obj_list.split(",")

    gaussians = model.extract(args)  
    
    print("Running segmentation...")
    select_gs_for_phys(
        gaussians,
        iteration,
        fg_obj_list,
        bg_obj_list,
        ground_plane_name,
        threshold,
        object_select_eps,
        inward_bbox_offset,
        final_noise_filtering,
        interactive_viz,
        rigid_object_name
    )

    
    #load pth file 
    import pickle
    edit_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "editing_modifier.pkl")
    with open(edit_path, "rb") as f:
        editing_dict = pickle.load(f)


    #Store objects
    object_masks = {}
    for obj_entry in editing_dict['objects']:
        name = obj_entry['name']
        mask = obj_entry['affected_gaussian_idx']
        object_masks[name] = mask

        save_splat(gaussians.get_xyz[mask], gaussians.get_distill_features[mask],
                   os.path.join(output, f"{name}_splat.pth"))

        if save_ply_flag:
            save_ply(gaussians.get_xyz[mask], os.path.join(output, f"{name}.ply"))

    #Create zeroed out mask
    total_mask = torch.zeros(gaussians.get_xyz.shape[0], dtype=bool)
    
    #Create room splat
    for mask in object_masks.values():
        total_mask |= mask
    room_mask = ~total_mask

    save_splat(gaussians.get_xyz[room_mask], gaussians.get_distill_features[room_mask],
            os.path.join(output, "room_splat.pth"))
    if save_ply_flag:
        save_ply(gaussians.get_xyz[room_mask], os.path.join(output, "room.ply"))


    print(f"Export complete. Saved splats in {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--fg_obj_list", type=str, required=True)
    parser.add_argument("--bg_obj_list", type=str, default="")
    parser.add_argument("--ground_plane_name", type=str, required=True)
    parser.add_argument("--rigid_object_name", type=str, default="")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--object_select_eps", type=float, default=0.1)
    parser.add_argument("--inward_bbox_offset", type=float, default=0.1)
    parser.add_argument("--final_noise_filtering", action="store_true")
    parser.add_argument("--interactive_viz", action="store_true")
    parser.add_argument("--output_dir", type=str, default="segmented_splats")
    parser.add_argument("--save_ply", action="store_true")
    args = parser.parse_args()

    fg_list = args.fg_obj_list.split(",")
    bg_list = args.bg_obj_list.split(",") if args.bg_obj_list else []

    extract_objects(
        model_path=args.model_path,
        iteration=args.iteration,
        fg_obj_list=fg_list,
        bg_obj_list=bg_list,
        ground_plane_name=args.ground_plane_name,
        rigid_object_name=args.rigid_object_name,
        threshold=args.threshold,
        object_select_eps=args.object_select_eps,
        inward_bbox_offset=args.inward_bbox_offset,
        final_noise_filtering=args.final_noise_filtering,
        interactive_viz=args.interactive_viz,
        output=args.output_dir,
        save_ply_flag=args.save_ply
    )


