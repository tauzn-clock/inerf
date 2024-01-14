import torch
from torch import tensor
import numpy as np
from copy import deepcopy
from pathlib import Path
import os
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

    camera = trainer.pipeline.datamanager.train_dataparser_outputs.cameras.camera_to_worlds

    correction = trainer.pipeline.model.camera_optimizer.forward([0]).detach()
    camera = camera.to(correction.device)

    corrected_pose = correct_pose(camera, correction)
    corrected_pose = corrected_pose.to("cpu")

    corrected_pose = transform_poses_to_original_space(
        corrected_pose,
        trainer.pipeline.datamanager.train_dataparser_outputs.dataparser_transform,
        trainer.pipeline.datamanager.train_dataparser_outputs.dataparser_scale,
        "opengl"
    )

    return corrected_pose

def eval_image(pipeline, transforms, eval_path):
    
    data = transforms["frames"]
    
    print(pipeline.datamanager.train_dataparser_outputs.image_filenames)
    
    custom_train_dataparser_outputs = pipeline.datamanager.train_dataparser_outputs
    custom_train_dataparser_outputs.image_filenames = []
    custom_train_dataparser_outputs.mask_filenames = []
    
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
        tf = np.asarray(data[i]["transform_matrix"])
        tf = tf[:3, :]
        camera_to_worlds = torch.cat([camera_to_worlds, tensor([tf]).float()], 0)   
    
    custom_cameras = pipeline.datamanager.train_dataparser_outputs.cameras
    custom_cameras.camera_to_worlds = transform_original_space_to_pose(camera_to_worlds,
                                                                        custom_train_dataparser_outputs.dataparser_transform,
                                                                        custom_train_dataparser_outputs.dataparser_scale,
                                                                        "opengl")
    custom_cameras.fx = fx
    custom_cameras.fy = fy
    custom_cameras.cx = cx
    custom_cameras.cy = cy
    custom_cameras.distortion_params = distortion_params
    custom_cameras.height = height
    custom_cameras.width = width
    custom_cameras.camera_type = camera_type
    custom_cameras.camera_to_worlds = camera_to_worlds
    custom_train_dataparser_outputs.cameras = custom_cameras
        
    pipeline.datamanager.train_dataparser_outputs = custom_train_dataparser_outputs
    pipeline.datamanager.train_dataset = pipeline.datamanager.create_train_dataset()
    pipeline.datamanager.setup_train()