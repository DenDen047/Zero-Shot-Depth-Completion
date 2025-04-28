#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
import torch
from loguru import logger
from PIL import Image

from utils.camera_geometry import depth_image_to_point_cloud
from utils.data_loader import load_calibration_json


def load_depth_map(depth_path: str) -> np.ndarray:
    """Load depth map from file.

    Parameters
    ----------
    depth_path : str
        Path to the depth map file

    Returns
    -------
    np.ndarray
        Depth map as numpy array
    """
    depth_img = Image.open(depth_path)
    depth_map = np.array(depth_img, dtype=np.float32)
    return depth_map


def load_rgb_image(rgb_path: str) -> np.ndarray:
    """Load RGB image from file.

    Parameters
    ----------
    rgb_path : str
        Path to the RGB image file

    Returns
    -------
    np.ndarray
        RGB image as numpy array
    """
    rgb_img = Image.open(rgb_path)
    return np.array(rgb_img)


def visualize_pointcloud(
    pcd: o3d.geometry.PointCloud,
    window_name: str = "Depth Point Cloud",
    save_path: Optional[str] = None,
) -> None:
    """Visualize point cloud using Open3D.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
        Point cloud to visualize
    window_name : str, optional
        Name of the visualization window, by default "Depth Point Cloud"
    save_path : Optional[str], optional
        Path to save the visualization as an image, by default None
    """
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)

    # Add point cloud
    vis.add_geometry(pcd)

    # Set up camera
    ctr = vis.get_view_control()
    ctr.set_zoom(0.5)

    # change background to grey
    opt = vis.get_render_option()
    opt.background_color = np.array([0.5, 0.5, 0.5])

    # Save visualization if requested
    if save_path is not None:
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_path)

    # Run visualization
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize depth map as 3D point cloud"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input directory containing depth maps and calibration",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save visualization images",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=80.0,
        help="Maximum depth value to consider",
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = (
        Path(args.output_dir) if args.output_dir else input_dir / "visualizations"
    )
    output_dir.mkdir(exist_ok=True)

    # Load camera parameters
    calibration_path = input_dir / "manual_calibration.json"
    if calibration_path.exists():
        logger.info(f"Loading camera parameters from {calibration_path}")
        calibration = load_calibration_json(str(calibration_path))
        intrinsics_K = calibration["intrinsics_K"]
        extrinsics_Rt = calibration["extrinsics_Rt"]
    else:
        logger.error("No calibration file found")
        raise ValueError("No calibration file found")

    # Load depth map and RGB image
    depth_path = input_dir / "depth_estimation_output" / "marigold_pred_raw.png"
    rgb_path = input_dir / "rgb.jpeg"

    logger.info(f"Loading depth map from {depth_path}")
    depth_map = load_depth_map(str(depth_path))

    logger.info(f"Loading RGB image from {rgb_path}")
    rgb_image = load_rgb_image(str(rgb_path))

    # Resize depth map to match RGB image size (use nearest neighbor to avoid interpolation artifacts)
    if depth_map.shape != rgb_image.shape[:2]:
        logger.warning(
            f"Depth map shape {depth_map.shape} does not match RGB image shape {rgb_image.shape[:2]}. Resizing depth map."
        )
        depth_img_pil = Image.fromarray(depth_map)
        depth_img_resized = depth_img_pil.resize(
            (rgb_image.shape[1], rgb_image.shape[0]), resample=Image.NEAREST
        )
        depth_map = np.array(depth_img_resized)

    # Convert to point cloud
    logger.info("Converting depth map to point cloud")
    np_pcd = (
        depth_image_to_point_cloud(
            torch.from_numpy(depth_map).unsqueeze(0),
            torch.from_numpy(rgb_image).permute(2, 0, 1),
            torch.from_numpy(intrinsics_K),
            torch.from_numpy(extrinsics_Rt),
        )
        .to("cpu")
        .numpy()
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(np_pcd[:, 3:] / 255.0)
    # visualize and output ply
    o3d.io.write_point_cloud(str(output_dir / "point_cloud.ply"), pcd)
    visualize_pointcloud(pcd, save_path=str(output_dir / "point_cloud.png"))


if __name__ == "__main__":
    main()
