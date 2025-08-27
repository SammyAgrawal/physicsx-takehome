import trimesh
from typing import Any
import numpy as np

class BaseDataTransform:
    # represents a generic data pre processing transform that users can extend and apply to their use case
    def __init__(self):
        pass

    def __call__(self, data: trimesh.PointCloud) -> Any:
        raise NotImplementedError

class ComposeTransforms(BaseDataTransform):
    def __init__(self, transforms: list[BaseDataTransform]):
        self.transforms = transforms
    
    def __call__(self, data: trimesh.PointCloud) -> Any:
        for transform in self.transforms:
            data = transform(data)
        return data

class Normalize(BaseDataTransform):
    def __init__(self, mu=None, sigma=None):
        self.mu = mu
        self.sigma = sigma
    
    def __call__(self, data: trimesh.PointCloud) -> Any:
        if self.mu is None:
            mu = data.vertices.mean(axis=0)
        if sigma is None:
            sigma = data.vertices.std(axis=0)
        return (data.vertices - mu) / sigma
    
    def set_mu_sigma(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma



class AffineTransfor3D(BaseDataTransform):
    def __init__(self, transform_matrix: np.array = None, method: str = "fixed"):
        self.transform_matrix = None
        if transform_matrix is not None:
            self.set_transform_matrix(transform_matrix)
    
    def set_transform_matrix(self, transform_matrix: np.array):
        assert transform_matrix.shape == (4, 4), "AffineTransfor3D only supports 3D homogeneous transformation matrices"
        transform_matrix[3,:3] = 0
        transform_matrix[3,3] = 1
        self.transform_matrix = transform_matrix

    
    def __call__(self, data: trimesh.PointCloud) -> Any:
        assert self.transform_matrix is not None, "AffineTransfor3D must be initialized with a transform matrix"
        if self.method == "random":
            self.set_random_transform()
        vertices = data.vertices
        assert vertices.shape[1] == 3, "AffineTransfor3D only supports 3D point clouds"
        N = vertices.shape[0]
        vertices = np.hstack([vertices, np.ones((N, 1))])
        transformed_vertices = vertices @ self.transform_matrix.T
        return transformed_vertices[:, :3]

    def make_transform_matrix(self, roll, pitch, yaw, tx, ty, tz):
        mat = np.eye(4)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])
        mat[:3, :3] = Rz @ Ry @ Rx
        mat[:3, 3] = [tx, ty, tz]
        mat[3, 3] = 1
        self.set_transform_matrix(mat)
    
    def set_random_transform(self):
        roll = np.random.uniform(-np.pi, np.pi)
        pitch = np.random.uniform(-np.pi, np.pi)
        yaw = np.random.uniform(-np.pi, np.pi)
        tx = np.random.uniform(-10, 10)
        ty = np.random.uniform(-10, 10)
        tz = np.random.uniform(-10, 10)
        self.make_transform_matrix(roll, pitch, yaw, tx, ty, tz)
        