import torch
from torch import tensor
import numpy as np
from copy import deepcopy
from pathlib import Path
import os
import json
import cv2
from scipy.spatial.transform import Rotation 
from nerfstudio.data.dataparsers.base_dataparser import transform_poses_to_original_space
from plane_nerf.plane_nerf_utils import transform_original_space_to_pose

def correct_pose(given_pose, correction):
    """Correct the given pose by the correction.

    Args:
        given_pose: The given pose.
        correction: The correction.

    Returns:
        The corrected pose.
    """

    given_pose = torch.cat(
        (
            given_pose,
            torch.tensor([[[0, 0, 0, 1]]], dtype=given_pose.dtype, device=correction.device).repeat_interleave(len(given_pose), 0),
        ),
        1,
    )

    correction = torch.cat(
        (
            correction,
            torch.tensor([[[0, 0, 0, 1]]], dtype=correction.dtype, device=correction.device).repeat_interleave(len(correction), 0),
        ),
        1,
    )

    corrected_pose = torch.matmul(correction,given_pose)

    return corrected_pose[:, :3, :4]

def get_corrected_pose(trainer):
    """Get the corrected pose.

    Args:
        trainer: The trainer.

    Returns:
        The corrected pose.
    """

    camera = trainer.pipeline.datamanager.train_dataparser_outputs.cameras.camera_to_worlds.to(trainer.pipeline.device)

    correction = trainer.pipeline.model.camera_optimizer.forward([i for i in range(trainer.pipeline.datamanager.train_dataparser_outputs.cameras.camera_to_worlds.shape[0])]) #WARNING: We are only getting the first pose

    corrected_pose = correct_pose(camera, correction)

    corrected_pose = transform_poses_to_original_space(
        corrected_pose,
        trainer.pipeline.datamanager.train_dataparser_outputs.dataparser_transform.to(trainer.pipeline.device),
        trainer.pipeline.datamanager.train_dataparser_outputs.dataparser_scale,
        "opengl"
    )

    return corrected_pose

def load_eval_image_into_pipeline(pipeline, eval_path, transform_file=None, starting_pose=None):
    
    TRANSFORM_PATH = os.path.join(eval_path, "transforms.json")
    if transform_file is not None:
        TRANSFORM_PATH = os.path.join(eval_path, transform_file)
    with open(TRANSFORM_PATH) as f:
        transforms = json.load(f)

    
    data = transforms["frames"]
        
    custom_train_dataparser_outputs = pipeline.datamanager.train_dataparser_outputs
    custom_train_dataparser_outputs.image_filenames = []
    custom_train_dataparser_outputs.mask_filenames = []
    custom_train_dataparser_outputs.mask_midpt = []
    
    camera_to_worlds = tensor([]).float()
    fx = torch.stack([pipeline.datamanager.train_dataparser_outputs.cameras.fx[0]]*len(data),0)
    fy = torch.stack([pipeline.datamanager.train_dataparser_outputs.cameras.fy[0]]*len(data),0)
    cx = torch.stack([pipeline.datamanager.train_dataparser_outputs.cameras.cx[0]]*len(data),0)
    cy = torch.stack([pipeline.datamanager.train_dataparser_outputs.cameras.cy[0]]*len(data),0)
    distortion_params = torch.stack([pipeline.datamanager.train_dataparser_outputs.cameras.distortion_params[0]]*len(data),0)
    height = torch.stack([pipeline.datamanager.train_dataparser_outputs.cameras.height[0]]*len(data),0)
    width = torch.stack([pipeline.datamanager.train_dataparser_outputs.cameras.width[0]]*len(data),0)
    camera_type = torch.stack([pipeline.datamanager.train_dataparser_outputs.cameras.camera_type[0]]*len(data),0)
    
    for i in range(len(data)):
        custom_train_dataparser_outputs.image_filenames.append(Path(os.path.join(eval_path,data[i]["file_path"])).as_posix())
        custom_train_dataparser_outputs.mask_filenames.append(Path(os.path.join(eval_path,data[i]["mask_path"])).as_posix())
        mask = cv2.imread(custom_train_dataparser_outputs.mask_filenames[-1], cv2.IMREAD_GRAYSCALE)
        custom_train_dataparser_outputs.mask_midpt.append(get_mask_midpt(mask))
        if starting_pose is None:
            tf = np.asarray(data[i]["transform_matrix"])
        else:
            tf = starting_pose[i]
        tf = tf[:3, :]
        camera_to_worlds = torch.cat([camera_to_worlds, tensor([tf]).float()], 0)   
    
    custom_train_dataparser_outputs.mask_midpt = torch.tensor(custom_train_dataparser_outputs.mask_midpt).float()

    custom_cameras = pipeline.datamanager.train_dataparser_outputs.cameras
    custom_cameras.camera_to_worlds = transform_original_space_to_pose(camera_to_worlds,
                                                                        pipeline.datamanager.train_dataparser_outputs.dataparser_transform,
                                                                        pipeline.datamanager.train_dataparser_outputs.dataparser_scale,
                                                                        "opengl")
    custom_cameras.fx = fx
    custom_cameras.fy = fy
    custom_cameras.cx = cx
    custom_cameras.cy = cy
    custom_cameras.distortion_params = distortion_params
    custom_cameras.height = height
    custom_cameras.width = width
    custom_cameras.camera_type = camera_type
    custom_train_dataparser_outputs.cameras = custom_cameras
        
    pipeline.datamanager.train_dataparser_outputs = custom_train_dataparser_outputs
    pipeline.datamanager.train_dataset = pipeline.datamanager.create_train_dataset()
    pipeline.datamanager.setup_train()
    
    
    return pipeline


def get_relative_pose(ground_truth_poses, target_poses):
    """Get the relative pose.

    Args:
        ground_truth_poses: The ground truth poses.
        target_poses: The target poses.

    Returns:
        The relative pose.
    """
    
    dtype = ground_truth_poses.dtype
    device = ground_truth_poses.device
    
    ground_truth_4x4 = torch.cat(
        (
            ground_truth_poses,
            torch.tensor([[[0, 0, 0, 1]]], dtype = dtype, device = device).repeat_interleave(len(ground_truth_poses), 0),
        ),
        1,
    )

    R_inv = target_poses[:, :3, :3].transpose(1,2)
    t_inv = torch.matmul(target_poses[:, :3, :3].transpose(1,2), -target_poses[:, :3, 3].unsqueeze(-1))
    # Concat R_inv and t_inv   
    target_4x4 = torch.cat([R_inv, t_inv], dim=2)

    target_4x4 = torch.cat(
        (
            target_4x4,
            torch.tensor([[[0, 0, 0, 1]]], dtype = dtype, device = device).repeat_interleave(len(target_poses), 0),
        ),
        1, 
    )
    
    relative_pose = torch.matmul(ground_truth_4x4, target_4x4)
    
    return relative_pose

def get_absolute_diff_for_pose(pose):
    """Get the absolute difference for the pose.

    Args:
        pose: The pose.

    Returns:
        The absolute difference for the pose.
    """

    translation = pose[:, :3, 3]
    rotation = pose[:, :3, :3]
    
    translation_diff = torch.norm(translation, dim=1)
    
    trace = rotation.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    angle_rot = torch.acos((trace- 1) / 2)

    return translation_diff, angle_rot

def get_image(pipeline, pose):
    camera = pipeline.datamanager.train_dataparser_outputs.cameras
    camera.camera = transform_original_space_to_pose(pose,
                                            pipeline.datamanager.train_dataparser_outputs.dataparser_transform,
                                            pipeline.datamanager.train_dataparser_outputs.dataparser_scale,
                                            "opengl")
    outputs = pipeline.model.get_outputs_for_camera(camera=camera)    
    return outputs
                                           
def get_mask_midpt(mask):
    """Get the mask mid point of a numpy mask.

    Args:
        mask: The mask.

    Returns:
        The mask mid point.
    """
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours are found
    if len(contours) > 0:
        # Find the centroid of the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        
        # Calculate the centroid coordinates
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy
    return None

def get_origin(pose, camera):
    """Get the pixel coordinate of a origin.

    Args:
        pose: The pose.
        camera: Camera intrinsics

    Returns:
        The pixel coordinate of the orgin.
    """
    transform = np.array(pose).reshape(4,4)
    inv_transform = np.linalg.inv(transform)

    #Rotate about x axis by 180 degrees
    rot = Rotation.from_euler('x', 180, degrees=True)
    rot_matrix = rot.as_matrix()
    rot_matrix = np.pad(rot_matrix, (0,1), 'constant')
    rot_matrix[3,3] = 1
    inv_transform = rot_matrix @ inv_transform


    f = camera["focal_length"]
    rho_w = camera["pixel_width"]
    rho_h = camera["pixel_height"]
    u_0 = camera["cx"]
    v_0 = camera["cy"]

    #Intrinsic matrix
    intrinsic = [
        [f/rho_w, 0, u_0, 0],
        [0, f/rho_h, v_0, 0],
        [0, 0, 1, 0]
    ]
    intrinsic = np.array(intrinsic)


    plane_index = intrinsic @ inv_transform
    plane_index = plane_index[:,3]
    origin_coord = plane_index[:2]/plane_index[2]
    return origin_coord