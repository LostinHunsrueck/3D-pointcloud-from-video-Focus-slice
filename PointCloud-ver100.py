import cv2
import numpy as np
import open3d as o3d

def video_to_calibrated_point_cloud(
    video_path, 
    output_ply_path, 
    known_thickness_um=25.0, 
    top_focus_frame=1, 
    bottom_focus_frame=80,
    variance_threshold=15.0
):
    """
    Processes a black-and-white focus-stack video into a calibrated 3D Point Cloud.
    Calculates Z-step size dynamically based on a known biological reference thickness.
    """
    # 1. Read and ingest video frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frames = []
    print("Reading frames from video clip...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Process as grayscale since the source footage is black and white
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()

    num_frames = len(frames)
    if num_frames == 0:
        print("Error: No frames extracted.")
        return
        
    height, width = frames[0].shape
    print(f"Extracted {num_frames} frames. Resolution: {width}x{height}")

    # 2. Dynamic Hardware Calibration Hook
    frames_in_layer = abs(bottom_focus_frame - top_focus_frame)
    if frames_in_layer == 0:
        print("Warning: Top and bottom frames match. Defaulting Z-scaling factor to 1.0.")
        z_step = 1.0
    else:
        z_step = known_thickness_um / frames_in_layer
        print(f"--- Calibration Success ---")
        print(f"Specimen spans {frames_in_layer} frames over {known_thickness_um} µm.")
        print(f"Calculated Z-step step size: {z_step:.4f} µm per frame.\n")

    # 3. Trackers for the sharpest state per individual pixel coordinate
    max_laplacian = np.zeros((height, width), dtype=np.float32)
    best_z_frame = np.zeros((height, width), dtype=np.int32)
    best_intensity = np.zeros((height, width), dtype=np.uint8)

    print("Analyzing spatial sharpness peak across focus timeline...")
    for z, frame in enumerate(frames):
        # Calculate local edge contrast via Laplacian filter
        laplacian = cv2.Laplacian(frame, cv2.CV_64F, ksize=3)
        laplacian_abs = np.abs(laplacian)

        # Retain pixels only at their highest recorded sharpness (temporal isolation)
        mask = laplacian_abs > max_laplacian
        max_laplacian[mask] = laplacian_abs[mask]
        best_z_frame[mask] = z
        best_intensity[mask] = frame[mask]

    # Filter out points that fall below our contrast variance threshold
    # This aggressively discards out-of-focus blurs and uniform background gaps
    valid_mask = max_laplacian > variance_threshold

    # Isolate coordinates for the narrow points that survived
    y_indices, x_indices = np.where(valid_mask)
    
    points = []
    colors = []

    print(f"Extracting coordinates for {len(x_indices)} sharp boundary points...")
    for y, x in zip(y_indices, x_indices):
        # Map pixel space to true 3D spatial coordinates using calibration factor
        z_coord = best_z_frame[y, x] * z_step
        points.append([x, y, z_coord])
        
        # Match gray value mapping for the 3D viewer
        intensity = best_intensity[y, x] / 255.0
        colors.append([intensity, intensity, intensity])

    if len(points) == 0:
        print("Warning: No points met the threshold. Lower VARIANCE_THRESHOLD.")
        return

    # 4. Generate Open3D Structures
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    # Apply Statistical Outlier Removal to eliminate stray floaters from optical noise
    print("Cleaning noise floor via statistical outlier filtering...")
    pcd_filtered, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)

    # Save the 3D model file
    o3d.io.write_point_cloud(output_ply_path, pcd_filtered)
    print(f"Success! Scaled 3D Point cloud saved to: {output_ply_path}")

    # Launch the interactive hardware-accelerated viewer window
    print("Launching Interactive 3D Viewer. Close the window to exit.")
    o3d.visualization.draw_geometries([pcd_filtered], 
                                      window_name="Calibrated Subtractive Focus Cloud",
                                      width=1920, height=1080)

if __name__ == "__main__":
    # --- FILE SETUP ---
    VIDEO_FILE = "record_mode1_2026-10-15_22-36-32.wmv"   # Path to your focus video file
    OUTPUT_MODEL = "3dPointcloud-Stomata.ply" # Output 3D data file
    
    # --- USER BIOLOGICAL CALIBRATION PARAMETERS ---
    KNOWN_THICKNESS = 1500.0       # Average physical thickness of your specimen layer in micrometers (µm)
    START_FRAME_OF_CELL = 1     # Video frame index where top layer snaps into sharp focus
    END_FRAME_OF_CELL = 550       # Video frame index where bottom layer snaps into sharp focus
    
    # --- FILTER SENSITIVITY ---
    # Increase to drop more blur; decrease if structural boundaries are vanishing entirely
    FINE_EDGE_THRESHOLD = 2.0            

    # Run execution pipeline
    video_to_calibrated_point_cloud(
        video_path=VIDEO_FILE,
        output_ply_path=OUTPUT_MODEL,
        known_thickness_um=KNOWN_THICKNESS,
        top_focus_frame=START_FRAME_OF_CELL,
        bottom_focus_frame=END_FRAME_OF_CELL,
        variance_threshold=FINE_EDGE_THRESHOLD
    )
