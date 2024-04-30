"""Generate superpoint Scanet cluster data for Scanet."""

import os
import sys
import pickle

# Add the project's files to the python path
# file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # for .py script
file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(file_path, "external", "superpoint_transformer")
sys.path.append(file_path)

# Necessary for advanced config parsing with hydra and omegaconf
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval)

import hydra
from src.utils import init_config
import torch
from src.visualization import show
from src.datasets.s3dis import CLASS_NAMES, CLASS_COLORS
from src.datasets.s3dis import S3DIS_NUM_CLASSES as NUM_CLASSES
from src.transforms import *

from tqdm import tqdm
import logging
import open3d as o3d
import pickle
import numpy as np
from src.data import NAG, Data
from scipy.spatial.transform import Rotation as R

log = logging.getLogger(__name__)


def visualize_superpoint(superpoint_data):
    pos = superpoint_data["pos"]
    normal = superpoint_data["normal"]
    super_indexes = superpoint_data["super_index"]
    num_color = np.max(super_indexes[0]) + 1
    # Generate random color
    color = np.random.rand(num_color, 3)
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    for i, super_index in enumerate(super_indexes):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pos)
        # pcd.normals = o3d.utility.Vector3dVector(normal)
        pcd.colors = o3d.utility.Vector3dVector(color[super_index])
        o3d.visualization.draw_geometries([pcd, origin])


def has_outlier(points, range: float = 3.0):
    """Check if there are outliers in the point cloud."""
    mean = np.mean(points, axis=0)
    # Check if there exists a point that is outside the range
    dist = np.linalg.norm(points - mean, axis=1)
    return np.any(dist > range)


def check_pcd(points, colors, normals):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([pcd])


def downsample_points(points, colors, normals, voxel_size: float = 0.03):
    pcd = o3d.t.geometry.PointCloud()
    pcd.point["positions"] = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
    if colors is not None:
        pcd.point["colors"] = o3d.core.Tensor(colors, dtype=o3d.core.Dtype.UInt8)
    if normals is not None:
        pcd.point["normals"] = o3d.core.Tensor(normals, dtype=o3d.core.Dtype.Float32)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    points = pcd.point["positions"].numpy()
    if colors is not None:
        colors = pcd.point["colors"].numpy()
    else:
        colors = np.zeros_like(points).astype(np.uint8)
    if normals is not None:
        normals = pcd.point["normals"].numpy()
    return points, colors, normals


class SuperPointTool:
    def __init__(self, pre_transform=None, **kwargs):
        self.pre_transform = pre_transform
        self.kwargs = kwargs

    def wrap_data(self, points: np.ndarray, colors: np.ndarray, normals: np.ndarray, **kwargs):
        """Wrap data into a PyTorch tensor."""
        # label
        pos = torch.from_numpy(points).to(torch.float32)
        # Compute the intensity from the RGB values
        cloud_colors = colors / 255.0
        intensity = cloud_colors[:, 0] * 0.299 + cloud_colors[:, 1] * 0.587 + cloud_colors[:, 2] * 0.114
        intensity = torch.from_numpy(intensity).unsqueeze(1).to(torch.float32)
        rgb = torch.from_numpy(cloud_colors).to(torch.float32)
        normals = torch.from_numpy(normals).to(torch.float32)
        custom_dict = {}
        for k, v in kwargs.items():
            custom_dict[f"raw_{k}"] = torch.from_numpy(v).to(torch.float32)
        data = Data(pos=pos, intensity=intensity, rgb=rgb, raw_normal=normals, **custom_dict)
        return data

    def preprocess(self, data: Data):
        # Apply pre_transform
        if self.pre_transform is not None:
            nag = self.pre_transform(data)
        else:
            nag = NAG([data])
        return nag

    def gen_superpoint(self, points: np.ndarray, colors: np.ndarray, normals: np.ndarray, scale: float = 1.0, vis: bool = False, **kwargs):
        """Generate superpoint data from input points and colors."""
        points = points * scale  # Scale the points
        # Move points to the origin
        points_center = np.mean(points, axis=0)
        points -= points_center
        data = self.wrap_data(points, colors, normals, **kwargs)
        nag = self.preprocess(data)

        # Construct superpoint data for parent object
        pos = (nag[0].pos.detach().cpu().numpy() + points_center) / scale
        color = nag[0].rgb.detach().cpu().numpy()
        normal = nag[0].raw_normal.detach().cpu().numpy()
        # normal = nag[0].normal.detach().cpu().numpy()
        planarity = nag[0].planarity.detach().cpu().numpy()
        linearity = nag[0].linearity.detach().cpu().numpy()
        verticality = nag[0].verticality.detach().cpu().numpy()
        scattering = nag[0].scattering.detach().cpu().numpy()
        custom_dict = {}
        for k, v in kwargs.items():
            custom_dict[k] = nag[0][f"raw_{k}"].detach().cpu().numpy()
        super_index_list = []
        for i in range(nag.num_levels - 1):
            _super_index = nag[i].super_index.detach().cpu().numpy()
            # Remap from last super_index to current super_index
            if i == 0:
                super_index = _super_index
            else:
                super_index = np.zeros_like(super_index_list[-1])
                for j in range(len(super_index_list[-1])):
                    super_index[j] = _super_index[super_index_list[-1][j]]
            super_index_list.append(super_index)
        super_point_data = {
            "pos": pos,
            "color": color,
            "normal": normal,
            "planarity": planarity,
            "linearity": linearity,
            "verticality": verticality,
            "scattering": scattering,
            "super_index": super_index_list,
            **custom_dict,
        }
        if vis:
            # Show the superpoint data
            show(nag, class_names=CLASS_NAMES, ignore=NUM_CLASSES, class_colors=CLASS_COLORS, max_points=100000)
        return super_point_data


if __name__ == "__main__":
    # Parse task cfg
    root_path = os.path.dirname((os.path.abspath(__file__)))
    scale = 1.0
    vis = False
    downsample_voxel_size = 0.02
    # Parse the configs using hydra
    cfg = init_config(
        overrides=[
            "experiment=semantic/scannet.yaml",
            "datamodule.voxel=0.03",
            "datamodule.pcp_regularization=[0.01, 0.1]",
            "datamodule.pcp_spatial_weight=[0.1, 0.1]",
            "datamodule.pcp_cutoff=[10, 10]",
            "datamodule.graph_gap=[0.2, 0.5]",
            "datamodule.graph_chunk=[1e6, 1e5]",
            "+net.nano=True",
        ]
    )
    # Instantiate the datamodule
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    # Initialize SuperPointTool
    spt = SuperPointTool(pre_transform=datamodule.pre_transform)

    data_dir = "/home/harvey/Data/processed_data/data"
    export_dir = "/home/harvey/Data/processed_data/output"
    os.makedirs(export_dir, exist_ok=True)
    super_point_dict = {}
    failed_lists = []
    for data_folder in tqdm(os.listdir(data_dir), desc="Processing Scannet data"):
        data_file = os.path.join(data_dir, data_folder, "scans", "processed-0.02.pkl")
        if not os.path.exists(data_file):
            continue
        with open(data_file, "rb") as f:
            _data = pickle.load(f)

        superpoint_data = spt.gen_superpoint(_data["points"], _data["colors"], np.zeros_like(_data["points"]), vis=vis)
        print(f"Points: {_data['points'].shape[0]}, Superpoints: {superpoint_data['pos'].shape[0]}")
        visualize_superpoint(superpoint_data)
