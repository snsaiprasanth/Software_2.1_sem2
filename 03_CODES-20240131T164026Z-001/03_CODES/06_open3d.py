#! D:\2024_WORKSHOP_IAAC\src\env_ws\Scripts\python.exe

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os

def save_pcd_as_ply(pcd, output_file_path, initial_name_file):
    # Export each segmented point cloud as a PLY file

    os.makedirs(output_directory, exist_ok=True)
    # i = 0
    # for pcd in range(pcd_gen):
    for i in range(len(pcd)):
        output_file_path = os.path.join(output_directory, f"{initial_name_file}_{i}.ply")
        print("output_file_path: ", output_file_path)
        print("segment: ", pcd[i])
        print("i: ", i)
        o3d.io.write_point_cloud(output_file_path, pcd[i], format='ply')
        print(f"Segment {i} exported to: {output_file_path}")
        # i += 1

# Point Cloud data preparation
# DATANAME = r"D:\2024_WORKSHOP_IAAC\src\data\appartment_cloud.ply"
DATANAME = r"D:\2024_WORKSHOP_IAAC\3D_model\output\it_02_RGB.ply"
pcd = o3d.io.read_point_cloud(DATANAME)

# For getting the center of the Point Cloud
pcd_center = pcd.get_center()
pcd.translate(-pcd_center)

nn = 16
std_multiplier = 10 # Standart multiplier 
filtered_pcd = pcd.remove_statistical_outlier(nn, std_multiplier) # Filter the Outliers from the Point Cloud
outliers = pcd.select_by_index(filtered_pcd[1],invert=True)
outliers.paint_uniform_color([0,1,0]) # Paint the Outliers in Green

filtered_pcd = filtered_pcd[0] 
o3d.visualization.draw_geometries([filtered_pcd, outliers])

# Voxel downsampling
voxel_size = 0.05  # 0.01 = 1cm (Best candidate each centimeter)
pcd_downsampled = filtered_pcd.voxel_down_sample(voxel_size = voxel_size)
# o3d.visualization.draw_geometries([pcd_downsampled])

#! CHECK NORMALS
# Estimating normals
nn_distance = np.mean(pcd.compute_nearest_neighbor_distance())

radius_normals = nn_distance*4

pcd_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normals, max_nn=16), fast_normal_computation=True)

# o3d.visualization.draw_geometries([pcd_downsampled, outliers])

# Extracting and setting parameters 
# front= [ 0.89651771639773403, 0.31412342382693859, 0.31238191174943813 ]
# lookat= [ 0.14556967273346125, -0.17414860502989729, -0.12685262830586536 ]
# up=  [ -0.28745233633438866, -0.12406307178727026, 0.94972601762540343 ]
# zoom= 0.21999999999999975

front= [ 0.68977455466174031, 0.0033025001524361183, -0.72401668298040001 ]
lookat= [ -0.83064266762740968, -0.62608128618060976, -0.22659771898653167 ]
up=  [ 0.031111508450987563, 0.99893076227546862, 0.034196582017148126 ]
zoom=  0.080000000000000002



pcd= pcd_downsampled
# o3d.visualization.draw_geometries([pcd],zoom=zoom,front=front,lookat=lookat,up=up )

# # RANSAC Planar Segmentation

pt_to_plane_dist = 0.4 # distance limit for a outlier
# pt_to_plane_dist = 0.02

plane_model, inliers = pcd.segment_plane(distance_threshold=pt_to_plane_dist, ransac_n=3, num_iterations=1000) # ransac_n == minimium number of points to create a plane
[a, b, c, d] = plane_model # planar equation

inlier_cloud = pcd.select_by_index(inliers)
outliers_cloud = pcd.select_by_index(inliers, invert=True)
inlier_cloud.paint_uniform_color([1,0,0])
outliers_cloud.paint_uniform_color([0,1,0])
# o3d.visualization.draw_geometries([inlier_cloud, outliers_cloud], zoom=zoom,front=front,lookat=lookat,up=up)

# Multi-order-ransac
max_plane_idx = 7 # Number of planes that will have

segment_models = {}
segments = {}
segments_color = {}
rest = pcd
'''
In the given code, `rest` is a variable that represents the remaining point cloud after segmenting
out the planes. It is initially set to the original point cloud `pcd`, and then in each iteration of
the loop, the inliers of the segmented plane are removed from `rest` using the `select_by_index`
function with the `invert=True` argument. This way, `rest` contains the points that do not belong to
any of the segmented planes.
'''
output_directory = r"D:\2024_WORKSHOP_IAAC\src\output" 

# initial_name_file = "outside_part_RGB"

for i in range(max_plane_idx):
    colors = plt.get_cmap('tab20')(i)
    segment_models[i], inliers = rest.segment_plane(distance_threshold=pt_to_plane_dist,ransac_n=3, num_iterations=1000)
    segments[i]=rest.select_by_index(inliers)
    # segments_color[i] = segments[i]
    # # save_pcd_as_ply(segments[i], output_directory, initial_name_file)
    segments_color[i].paint_uniform_color(list(colors[:3]))
    rest = rest.select_by_index(inliers, invert=True)


# #! SAVE PCs parts
# # Create a directory to save PLY files
# initial_name_file = "iaac_outside"
# save_pcd_as_ply(segments, output_directory, initial_name_file)
# initial_name_file = "iaac_outside_color"
# save_pcd_as_ply(segments_color, output_directory, initial_name_file)

# o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[rest], zoom=zoom,front=front,lookat=lookat,up=up)


# #! DBSCAN Clustering
# Uses core points to create the clusters (with the range for each point)
labels = np.array(rest.cluster_dbscan(eps=0.6, min_points=100))
# labels = np.array(rest.cluster_dbscan(eps=0.6, min_points=5))
max_label = labels.max() # To print hoy many cluester we have

# Create a list to store individual point clouds
clusters = []

for label in range(max_label + 1):
    if label == -1:
        continue  # Skip noise points
    colors = plt.get_cmap('tab20')(i)
    cluster_points = rest.select_by_index(np.where(labels == label)[0])
    
    # Append the cluster to the list
    clusters.append(cluster_points)

    # segment_models[i], inliers = rest.segment_plane(distance_threshold=pt_to_plane_dist,ransac_n=3, num_iterations=1000)
    # clusters[i]=rest.select_by_index(inliers)
    # # save_pcd_as_ply(segments[i], output_directory, initial_name_file)
    clusters[i].paint_uniform_color(list(colors[:3]))

# colors = plt.get_cmap('tab20')(labels / (max_label if max_label > 0 else 1))
# colors[labels < 0] = 0

# rest_color = rest
# # rest_color.colors = o3d.utility.Vector3dVector(colors[:, :3])


# o3d.visualization.draw_geometries([clusters],zoom=zoom,front=front,lookat=lookat,up=up )


# Filter out PointClouds with fewer than 1000 points
filtered_point_clouds = [pc for pc in clusters if len(pc.points) > 5]

initial_name_file = "iaac_inside_color"
save_pcd_as_ply(filtered_point_clouds, output_directory, initial_name_file)

# o3d.visualization.draw_geometries([rest_color],zoom=zoom,front=front,lookat=lookat,up=up )

# o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[rest], zoom=zoom,front=front,lookat=lookat,up=up)

