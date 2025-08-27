"""plot_point_cloud function implementation."""

import matplotlib.pyplot as plt
import trimesh


def plot_point_cloud(point_cloud: trimesh.PointCloud) -> None:
    """Plot a point cloud.

    Args:
        point_cloud: The point cloud to plot.
    """
    vertices = point_cloud.vertices if isinstance(point_cloud, trimesh.PointCloud) else point_cloud
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x, y, z, s=10, depthshade=True)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

    ax.set_aspect("equal")

    plt.show()
