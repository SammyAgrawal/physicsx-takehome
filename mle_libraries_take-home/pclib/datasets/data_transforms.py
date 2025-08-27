import trimesh
from typing import Any, List
import numpy as np

class BaseDataTransform:
    # represents a generic data pre processing transform that users can extend and apply to their use case
    def __init__(self):
        pass

    def __call__(self, data: trimesh.PointCloud) -> Any:
        # converts trimesh.PointCloud to numpy array
        return data.vertices


class Normalize(BaseDataTransform):
    def __init__(self, mu=None, sigma=None):
        self.mu = mu
        self.sigma = sigma
    
    def __call__(self, data: np.ndarray) -> Any:
        if isinstance(data, trimesh.PointCloud):
            data = data.vertices
        mu = self.mu if self.mu is not None else data.mean(axis=0)
        sigma = self.sigma if self.sigma is not None else data.std(axis=0)
        return (data - mu) / sigma
    
    def set_mu_sigma(self, mu, sigma):
        # global data set wide normalization, instead of per point cloud normalization
        self.mu = mu
        self.sigma = sigma


class AffineTransfor3D(BaseDataTransform):
    def __init__(self, transform_matrix: np.array = None, method: str = "fixed"):
        self.transform_matrix = None
        self.method = method
        if transform_matrix is not None:
            self.set_transform_matrix(transform_matrix)
    
    def set_transform_matrix(self, transform_matrix: np.array):
        assert transform_matrix.shape == (4, 4), "AffineTransfor3D only supports 3D homogeneous transformation matrices"
        transform_matrix[3,:3] = 0
        transform_matrix[3,3] = 1
        self.transform_matrix = transform_matrix

    
    def __call__(self, vertices: np.ndarray) -> Any:
        if isinstance(vertices, trimesh.PointCloud):
            vertices = vertices.vertices
        if self.method == "random":
            self.set_random_transform()
        assert self.transform_matrix is not None, "AffineTransfor3D must be initialized with a transform matrix"
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


class DownSample(BaseDataTransform):
    def __init__(self, voxel_size: List[float] = [1.0, 1.0, 1.0], method: str = "voxel", n_points: int = 1000):
        self.method = method
        self.set_voxel_size(voxel_size)
        self.n_points = n_points
    
    def set_voxel_size(self, voxel_size):
        if isinstance(voxel_size, float) or isinstance(voxel_size, int):
            self.voxel_size = np.array([voxel_size, voxel_size, voxel_size])
        elif isinstance(voxel_size, List):
            self.voxel_size = np.array(voxel_size)
    
    def __call__(self, data: np.ndarray) -> Any:
        if isinstance(data, trimesh.PointCloud):
            data = data.vertices
        if self.method == "voxel":
            return self.voxelize(data)
        else:
            raise ValueError(f"Invalid method: {self.method}")
    
    def voxelize(self, data):
        coords = np.floor(data / self.voxel_size)
        _, indices = np.unique(coords, axis=0, return_index=True)
        return data[indices]

class UniformSizedEmbedding(BaseDataTransform):
    def __init__(self, n_points: int = 1000, method: str = "random"):
        self.n_points = n_points
        self.method = method
    
    def __call__(self, data: trimesh.PointCloud) -> Any:
        if isinstance(data, trimesh.PointCloud):
            data = data.vertices
        N = data.shape[0]
        if N <= self.n_points:
            # pad with zero to fill out space
            padding = np.zeros((self.n_points - N, 3))
            data = np.vstack([data, padding])
            return data

        if self.method == "random":
            return data[np.random.choice(N, self.n_points, replace=False)]
        else:
            # would be cool to implement an autoencoder or gemetry based embedding method here. 
            raise ValueError(f"Invalid method: {self.method}")