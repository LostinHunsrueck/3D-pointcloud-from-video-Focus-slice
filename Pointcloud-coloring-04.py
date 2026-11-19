import os
import numpy as np
import open3d as o3d
from matplotlib.colors import LinearSegmentedColormap

input_path = "Your-Point-cloud-file.ply"
base, ext = os.path.splitext(input_path)
output_path = f"{base}-cbgy{ext}"

pcd = o3d.io.read_point_cloud(input_path)
pcd_tree = o3d.geometry.KDTreeFlann(pcd)
points = np.asarray(pcd.points)

# Smooth neighborhood search
mean_dist = np.mean(pcd.compute_nearest_neighbor_distance())
search_radius = mean_dist * 6.0

neighbor_counts = np.zeros(len(points))
for i, pt in enumerate(points):
    k, _, _ = pcd_tree.search_radius_vector_3d(pt, search_radius)
    neighbor_counts[i] = k

# Tighter upper clipping (5th to 92nd percentile) forces dense areas to hit maximum yellow
low, high = np.percentile(neighbor_counts, [5, 92])
density_clipped = np.clip(neighbor_counts, low, high)
density_norm = (density_clipped - low) / (high - low + 1e-8)

# Gamma < 1.0 boosts mid-to-high intensity so greens and yellows pop
density_norm = np.power(density_norm, 0.55)

# Intense biological gradient: Deep Cobalt Blue -> Vivid Emerald Green -> Luminous Yellow
bio_cmap = LinearSegmentedColormap.from_list(
    "bio_intense",
    [
        (0.0, "#0d3b66"),   # Low density: deep rich blue
        (0.68, "#2ea866"),  # Mid density: vivid biological green
        (1.0, "#fff06d")    # High density: intense bright yellow maximum
    ]
)

colors = bio_cmap(density_norm)[:, :3]
pcd.colors = o3d.utility.Vector3dVector(colors)

# Estimate normals for 3D depth shading
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=30))

o3d.io.write_point_cloud(output_path, pcd)
print(f"Saved colored point cloud to: {output_path}")

vis = o3d.visualization.Visualizer()
vis.create_window(window_name=f"Biological Density View - {os.path.basename(output_path)}")
vis.add_geometry(pcd)
render_opt = vis.get_render_option()
render_opt.background_color = np.asarray([0.04, 0.05, 0.07])
render_opt.point_size = 2.0
vis.run()
vis.destroy_window()
