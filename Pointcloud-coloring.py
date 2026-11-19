import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# Input file path
input_path = "your3d-pointcloud.ply"

# Build output path with '-c' added before the extension (e.g., wing_hairs-c.ply)
base, ext = os.path.splitext(input_path)
output_path = f"{base}-c5{ext}"

# Load point cloud
pcd = o3d.io.read_point_cloud(input_path)
pcd_tree = o3d.geometry.KDTreeFlann(pcd)
points = np.asarray(pcd.points)

# Automatically estimate a good search radius from average point spacing
mean_dist = np.mean(pcd.compute_nearest_neighbor_distance())
search_radius = mean_dist * 4

# Count neighbors within the 3D sphere for every point
neighbor_counts = np.zeros(len(points))
for i, pt in enumerate(points):
    k, _, _ = pcd_tree.search_radius_vector_3d(pt, search_radius)
    neighbor_counts[i] = k

# Log-scale normalization so dense sockets don't wash out fine hair tips
density_norm = np.log1p(neighbor_counts)
density_norm = (density_norm - density_norm.min()) / (density_norm.max() - density_norm.min() + 1e-8)

# Apply colormap ("inferno", "magma", "viridis", or "turbo")
colors = plt.get_cmap("inferno")(density_norm)[:, :3]
pcd.colors = o3d.utility.Vector3dVector(colors)

# Save the colored point cloud
o3d.io.write_point_cloud(output_path, pcd)
print(f"Saved colored point cloud to: {output_path}")

# Optional: Visualize with a dark background to match the simulation
vis = o3d.visualization.Visualizer()
vis.create_window(window_name=f"Density View - {os.path.basename(output_path)}")
vis.add_geometry(pcd)
vis.get_render_option().background_color = np.asarray([0.1, 0.1, 0.12])
vis.run()
vis.destroy_window()
