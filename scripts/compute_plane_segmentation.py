import argparse
import copy
import json
import os

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN, KMeans

DEBUG = False


def preprocess_pcd(scene_pcd):
    prev_points_n = len(scene_pcd.points)
    if DEBUG:
        o3d.visualization.draw_geometries([scene_pcd])
    scene_pcd = scene_pcd.voxel_down_sample(voxel_size=0.05)
    print(f"Removed {prev_points_n - len(scene_pcd.points)} points.")
    prev_points_n = len(scene_pcd.points)
    if DEBUG:
        o3d.visualization.draw_geometries([scene_pcd])
    scene_pcd, _ = scene_pcd.remove_statistical_outlier(nb_neighbors=200, std_ratio=2.0)
    print(f"Removed {prev_points_n - len(scene_pcd.points)} points.")
    if DEBUG:
        o3d.visualization.draw_geometries([scene_pcd])
    return scene_pcd


def save_visualization(geometries, filename, width=512, height=512):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    for geometry in geometries:
        vis.add_geometry(geometry)

    vis.get_render_option().background_color = [1.0, 1.0, 1.0]
    vis.update_renderer()
    vis.poll_events()
    vis.update_geometry(geometries[0])

    image = vis.capture_screen_float_buffer(do_render=True)
    cv2.imwrite(filename, np.array(image)[:, :, ::-1] * 255)
    vis.destroy_window()


def cluster_normals(pcd, angle_threshold=10, min_points=400, eps=0.1, min_samples=5, n_seeds=12):
    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)
    clusters = []
    clusters_avg_normals = []
    remaining_indices = set(range(len(normals)))
    noise_indices = set()

    vis_pcd = o3d.geometry.PointCloud(pcd)

    kmeans = KMeans(n_clusters=n_seeds, random_state=42)
    kmeans.fit(normals)
    seed_normals = kmeans.cluster_centers_

    if DEBUG:
        labels = kmeans.labels_
        vis_colors = np.array([np.random.rand(3) for _ in range(len(np.unique(labels)))])
        pcd_colors = np.zeros((len(normals), 3))
        for i, label in enumerate(labels):
            pcd_colors[i] = vis_colors[label]
        vis_pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
        o3d.visualization.draw_geometries([vis_pcd], window_name="Normal Clustering")

    for seed_normal in seed_normals:
        if len(remaining_indices) < min_points:
            break

        cluster = []
        for i in remaining_indices:
            angle = np.arccos(np.clip(np.dot(seed_normal, normals[i]), -1.0, 1.0)) * 180 / np.pi
            if np.abs(angle) <= angle_threshold:
                cluster.append(i)
        if len(cluster) >= min_points:
            cluster_points = points[cluster]

            cluster_dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = cluster_dbscan.fit_predict(cluster_points)

            unique_labels = set(cluster_labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)

            if len(unique_labels) == 0:
                print("DBSCAN failed to find subclusters. Treating as a single cluster.")
                subclusters = [cluster]
            else:
                subclusters = [np.array(cluster)[cluster_labels == label].tolist() for label in unique_labels]

            for subcluster in subclusters:
                if len(subcluster) >= min_points:
                    # Check if the bbox is close to mean
                    subcluster_points = points[subcluster]
                    bbox = np.array([np.min(subcluster_points, axis=0), np.max(subcluster_points, axis=0)])
                    bbox_center = np.mean(bbox, axis=0)

                    sub_pcd = vis_pcd.select_by_index(subcluster)
                    sub_pcd.paint_uniform_color([1, 0, 0])

                    bbox_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                    bbox_sphere.translate(bbox_center)
                    # bbox_sphere.paint_uniform_color([0, 1, 0])

                    mean_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                    mean_sphere.translate(np.mean(subcluster_points, axis=0))

                    if DEBUG:
                        print(f"abs(bbox - mean) {np.abs(np.linalg.norm(bbox_center - np.mean(subcluster_points, axis=0)))}")
                        o3d.visualization.draw_geometries([sub_pcd, bbox_sphere, mean_sphere],
                                                           window_name="Subcluster BBox and Mean Visualization",
                                                           width=800, height=600)

                    clusters.append(subcluster)
                    remaining_indices -= set(subcluster)

                    # Visualize the current subcluster
                    vis_colors = np.asarray(vis_pcd.colors)
                    subcluster_color = np.random.rand(3)
                    vis_colors[subcluster] = subcluster_color

                    subcluster_pcd = vis_pcd.select_by_index(subcluster)
                    subcluster_pcd.paint_uniform_color(subcluster_color)

                    clusters_avg_normals.append(np.mean(normals[subcluster], axis=0) / np.linalg.norm(np.mean(normals[subcluster], axis=0)))

                    other_pcd = vis_pcd.select_by_index(list(remaining_indices))
                    other_pcd.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray

                    if DEBUG:
                        o3d.visualization.draw_geometries([subcluster_pcd, other_pcd],
                                                           window_name="Subcluster Visualization",
                                                           width=800, height=600)
                        print(f"Visualizing subcluster with {len(subcluster)} points")
                else:
                    noise_indices.update(subcluster)

            print(f"Remaining indices for next iter: {len(remaining_indices)}")
        else:
            if len(cluster) > 0:
                print(f"Cluster with {len(cluster)} points is too small. Adding to noise.")
                noise_indices.update(cluster)
                remaining_indices -= set(cluster)

    if remaining_indices:
        noise_indices.update(remaining_indices)

    if noise_indices and clusters:
        print(f"Assigning {len(noise_indices)} noise points to closest clusters")
        tree = cKDTree([cluser_point for cluster in clusters for cluser_point in points[cluster]])
        tree_points_all_points_map = {}
        current_point_idx = 0
        for cluster in clusters:
            for point_idx in cluster:
                tree_points_all_points_map[current_point_idx] = point_idx
                current_point_idx += 1

        new_clusters = copy.deepcopy(clusters)
        for idx in noise_indices:
            _, point_idx = tree.query(points[idx])
            closest_cluster_idx = -1
            for i, cluster in enumerate(clusters):
                if tree_points_all_points_map[point_idx] in cluster:
                    closest_cluster_idx = i
                    break
            new_clusters[closest_cluster_idx].append(idx)
        clusters = new_clusters
        vis_colors = np.asarray(vis_pcd.colors)
        for i, cluster in enumerate(clusters):
            cluster_color = np.random.rand(3)
            vis_colors[cluster] = cluster_color

        if DEBUG:
            print("Visualizing final result with noise points assigned")
            o3d.visualization.draw_geometries([vis_pcd],
                                               window_name="Final Result",
                                               width=800, height=600)
    elif not clusters:
        # If no clusters were found, treat all points as a single cluster
        clusters = [list(range(len(normals)))]
        clusters_avg_normals = [np.mean(normals, axis=0)]
    return clusters, np.asarray(clusters_avg_normals)


def create_plane_mesh(plane_model, points, color):
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)

    u = np.array([1, 0, 0])
    if np.abs(np.dot(u, normal)) > 0.999:
        u = np.array([0, 1, 0])
    v = np.cross(normal, u)
    u = np.cross(v, normal)
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)

    projected_points = points - np.outer(np.dot(points, normal) + d, normal)

    u_coords = np.dot(projected_points, u)
    v_coords = np.dot(projected_points, v)

    min_u, max_u = np.min(u_coords), np.max(u_coords)
    min_v, max_v = np.min(v_coords), np.max(v_coords)

    extents = np.array([max_u - min_u, max_v - min_v])

    plane_mesh = o3d.geometry.TriangleMesh.create_box(width=extents[0], height=extents[1], depth=0.01)
    plane_mesh.compute_vertex_normals()
    plane_mesh.paint_uniform_color(color)

    center_3d = np.mean(points, axis=0)

    rotation = np.column_stack((u, v, normal))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = center_3d - rotation @ (extents[0]/2, extents[1]/2, 0.005)

    plane_mesh.transform(transform)

    debug_geometries = []

    original_points_pcd = o3d.geometry.PointCloud()
    original_points_pcd.points = o3d.utility.Vector3dVector(points)
    original_points_pcd.paint_uniform_color([1, 0, 0])

    projected_points_pcd = o3d.geometry.PointCloud()
    projected_points_pcd.points = o3d.utility.Vector3dVector(projected_points)
    projected_points_pcd.paint_uniform_color([0, 1, 0])

    center_pcd = o3d.geometry.PointCloud()
    center_pcd.points = o3d.utility.Vector3dVector([center_3d])
    center_pcd.paint_uniform_color([0, 0, 1])

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=min(extents)*0.5, origin=center_3d)

    normal_line = o3d.geometry.LineSet()
    normal_line.points = o3d.utility.Vector3dVector([center_3d, center_3d + normal * min(extents)*0.5])
    normal_line.lines = o3d.utility.Vector2iVector([[0, 1]])
    normal_line.colors = o3d.utility.Vector3dVector([[1, 1, 0]])

    debug_geometries = [original_points_pcd, projected_points_pcd, center_pcd, coordinate_frame, normal_line, plane_mesh]

    if DEBUG:
        o3d.visualization.draw_geometries(debug_geometries, window_name="Plane Mesh Debug Visualization")

    return plane_mesh

def segment_multiple_planes(pcd, distance_threshold=0.05, ransac_n=4, num_iterations=1000, min_points=200):
    clusters, _ = cluster_normals(pcd, angle_threshold=3, min_points=min_points)

    planes = []
    plane_colors = []
    pcds = []
    plane_models = []
    plane_meshes = []

    for cluster in clusters:
        cluster_pcd = pcd.select_by_index(cluster)

        plane_model, inliers = cluster_pcd.segment_plane(distance_threshold=distance_threshold,
                                                         ransac_n=ransac_n,
                                                         num_iterations=num_iterations)

        if len(inliers) < min_points:
            continue

        inlier_cloud = cluster_pcd.select_by_index(inliers)
        pcds.append(inlier_cloud)
        plane_models.append(plane_model)
        print(f"Found a plane with {len(inliers)} inliers.")

        plane_color = np.random.rand(3)
        inlier_cloud.paint_uniform_color(plane_color)

        planes.append(inlier_cloud)
        plane_colors.append(plane_color)

        plane_mesh = create_plane_mesh(plane_model, np.asarray(inlier_cloud.points), plane_color)
        plane_meshes.append(plane_mesh)

    all_plane_indices = set()
    for cluster in clusters:
        all_plane_indices.update(cluster)
    remaining_indices = set(range(len(pcd.points))) - all_plane_indices
    remaining_points = pcd.select_by_index(list(remaining_indices))

    geometries = [pcd] + plane_meshes
    if DEBUG:
        o3d.visualization.draw_geometries(geometries, window_name="Planes and Point Cloud")

    return planes, plane_colors, remaining_points, pcds, plane_models


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='./output')
    args = parser.parse_args()

    scene_name = args.img_path.split('/')[-1].split('.')[0]
    out_dir = f"{args.out_dir}/{scene_name}/arch"
    os.makedirs(out_dir, exist_ok=True)

    # import time
    # start = time.time()

    scene_id = out_dir.split("/")[-1]
    print(f"\nProcessing scene {scene_id}")
    scene_pcd = o3d.io.read_point_cloud(f"{out_dir}/inpainted_pcd.ply")
    processed_scene_pcd = preprocess_pcd(scene_pcd)
    point_to_pixel_map = np.load(f"{out_dir}/inpainted_pcd_mapping.npz", allow_pickle=True)["map"].tolist()

    if os.path.exists(f"{out_dir}/inpainted_pcd"):
        os.system(f"rm -r {out_dir}/inpainted_pcd")
    os.makedirs(f"{out_dir}/inpainted_pcd", exist_ok=True)
    clusters, cluster_avg_normals = cluster_normals(processed_scene_pcd, angle_threshold=10, min_points=200)
    # Determine which one is the floor plane (the most y-up normal and largest cluster)
    candidates = [(n[1], len(clusters[i])) for i, n in enumerate(cluster_avg_normals)]
    floor_idx = None
    current_candidate = None
    for i, candidate in enumerate(candidates):
        if candidate[0] > 0.7 and (current_candidate is None or current_candidate[1] < candidate[1]):
            # print(f"New floor candidate: {candidate} ({i}), better than {current_candidate} {floor_idx}")
            floor_idx = i
            current_candidate = candidate
    ceiling_ids = []
    ceiling_candidate_idx = np.argmin([n[1] for n in cluster_avg_normals])
    if cluster_avg_normals[ceiling_candidate_idx][1] < -0.7:
        ceiling_ids.append(ceiling_candidate_idx)
    else:
        ceiling_ids = None
    # Remove ceiling plane
    new_clusters = []
    new_cluster_avg_normals = []
    new_floor_idx = None
    if ceiling_ids is not None:
        for i, cluster in enumerate(clusters):
            if i not in ceiling_ids:
                new_clusters.append(cluster)
                new_cluster_avg_normals.append(cluster_avg_normals[i])
            if i == floor_idx:
                new_floor_idx = len(new_clusters) - 1
        clusters = new_clusters
        cluster_avg_normals = new_cluster_avg_normals
        floor_idx = new_floor_idx

    # Map segmentation (clusters) to original point cloud using KNN
    original_points = np.asarray(scene_pcd.points)
    processed_points = np.asarray(processed_scene_pcd.points)

    tree = cKDTree(processed_points)
    _, indices = tree.query(original_points)

    mapping = np.zeros(len(original_points), dtype=int)
    for i, cluster in enumerate(clusters):
        for idx in cluster:
            mapping[indices == idx] = i + 1  # Add 1 to avoid 0 as a label

    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd.points = o3d.utility.Vector3dVector(original_points)
    vis_colors = np.array([np.random.rand(3) for _ in range(max(mapping) + 1)])
    vis_pcd.colors = o3d.utility.Vector3dVector(vis_colors[mapping])

    if DEBUG:
        o3d.visualization.draw_geometries([vis_pcd])
    if os.path.exists(f"{out_dir}/inpainted_pcd/clusters"):
        os.system(f"rm -r {out_dir}/inpainted_pcd/clusters")
    os.makedirs(f"{out_dir}/inpainted_pcd/clusters", exist_ok=True)
    # Save clusters on full point clouds, remember to use mapping since clusters are on processed point cloud
    sem_ref = {}
    for i, cluster in enumerate(clusters):
        if i == floor_idx:
            pcd_name = f"wall_{i+1}"
            sem_ref[str(i + 1)] = "floor"
        else:
            pcd_name = f"wall_{i+1}"
            sem_ref[str(i + 1)] = "wall"
        cluster_pcd = scene_pcd.select_by_index(np.where(mapping == i + 1)[0])
        o3d.io.write_point_cloud(f"{out_dir}/inpainted_pcd/clusters/{pcd_name}.ply", cluster_pcd)
        np.save(f"{out_dir}/inpainted_pcd/clusters/{pcd_name}_normal.npy", cluster_avg_normals[i])
    os.makedirs(f"{out_dir}/inpainted_pcd/arch", exist_ok=True)
    json.dump(sem_ref, open(f"{out_dir}/inpainted_pcd/arch/semantic_reference.json", "w"))
    segmentation_image = np.zeros((1008, 784), dtype=int)
    for i, coord in point_to_pixel_map.items():
        segmentation_image[coord] = mapping[i]
    segmentation_image = segmentation_image.T
    cv2.imwrite(f"{out_dir}/inpainted_pcd/segmentation_image.png", segmentation_image)
    segmentation_image_vis = cv2.applyColorMap((segmentation_image / segmentation_image.max() * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(f"{out_dir}/inpainted_pcd/segmentation_image_vis.png", segmentation_image_vis)
    np.save(f"{out_dir}/inpainted_pcd/segmentation_image.npy", segmentation_image)
    print(f"Saved segmentation to {out_dir}/inpainted_pcd")
    # print(f"Time taken: {time.time() - start}")