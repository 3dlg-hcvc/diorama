import json
import math
import os
from glob import glob

import mapbox_earcut as earcut
import numpy as np
import open3d as o3d
import sympy as sp
import torch
import trimesh
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from tqdm import tqdm

allviz = []

def rotating_calipers(convex_hull_points):
    num_points = len(convex_hull_points)
    if num_points == 0:
        return None, None, None

    min_area = float('inf')
    best_rectangle = None
    rotation_angle = 0

    for i in range(num_points):
        # Define edge between point i and i+1
        p1 = convex_hull_points[i]
        p2 = convex_hull_points[(i + 1) % num_points]
        edge = p2 - p1

        # Calculate the angle to align the edge with the x-axis
        angle = -math.atan2(edge[1], edge[0])

        # Rotation matrix to align edge with x-axis
        rotation_matrix = np.array([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle),  math.cos(angle)]
        ])

        # Rotate all points
        rotated = np.dot(convex_hull_points, rotation_matrix)

        # Find the bounding rectangle in rotated space
        min_x, min_y = np.min(rotated, axis=0)
        max_x, max_y = np.max(rotated, axis=0)
        area = (max_x - min_x) * (max_y - min_y)

        if area < min_area:
            min_area = area
            rotation_angle = angle
            # Define rectangle corners in rotated space
            rect = np.array([
                [max_x, max_y],
                [min_x, max_y],
                [min_x, min_y],
                [max_x, min_y]
            ])
            best_rectangle = rect

    # Rotate rectangle corners back to original space
    inv_rotation_matrix = np.array([
        [math.cos(-rotation_angle), -math.sin(-rotation_angle)],
        [math.sin(-rotation_angle),  math.cos(-rotation_angle)]
    ])
    best_rectangle_rotated = np.dot(best_rectangle, inv_rotation_matrix)

    return min_area, best_rectangle_rotated, rotation_angle


def create_convex_hull_bounding_box(pcd, plane):
    # Extract points and project to 2D (x, y)
    points = np.asarray(pcd.points)
    normal = plane.arch_normal
    # Project to 2D
    centroid = np.mean(points, axis=0)

    # Project points to the plane
    projected_points = project_to_plane(points, plane.plane_model)

    # Create a coordinate system on the plane
    x_axis = np.array([1, 0, 0])
    if np.abs(np.dot(x_axis, normal)) > 0.9:
        x_axis = np.array([0, 1, 0])
    y_axis = np.cross(normal, x_axis)
    x_axis = np.cross(y_axis, normal)
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # Project points to 2D coordinate system
    points_2d = np.column_stack([
        np.dot(projected_points - centroid, x_axis),
        np.dot(projected_points - centroid, y_axis)
    ])

    # Compute convex hull
    try:
        hull = ConvexHull(points_2d)
    except Exception as e:
        raise ValueError(f"ConvexHull computation failed: {e}")

    hull_points = points_2d[hull.vertices]

    # Apply Rotating Calipers to find minimal bounding rectangle
    area, rectangle, angle = rotating_calipers(hull_points)

    if rectangle is None:
        raise ValueError("Rotating calipers failed to compute a bounding rectangle.")

    corners3d = (
        centroid +
        rectangle[:, 0][:, np.newaxis] * x_axis +
        rectangle[:, 1][:, np.newaxis] * y_axis
    )

    epsilon = 1e-6  # Tolerance for floating-point comparisons

    # Extract and normalize the current normal
    current_normal = plane.arch_normal
    norm = np.linalg.norm(current_normal)
    current_normal = current_normal / norm

    # Define target z-axis
    target_z = np.array([0, 0, 1])

    # Calculate rotation axis and angle
    rotation_axis = np.cross(current_normal, target_z)
    dot_product = np.dot(current_normal, target_z)
    rotation_angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    # Determine rotation matrix
    if np.linalg.norm(rotation_axis) < epsilon:
        if dot_product < -1 + epsilon:
            rotation_axis = np.array([1, 0, 0])
            rotation_angle = np.pi
            rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, rotation_axis)
        else:
            rotation_matrix = np.eye(4)
    else:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, rotation_axis)

    corners3d_rotated = np.dot(corners3d - centroid, rotation_matrix[:3, :3].T) + centroid

    # Visualization for debugging
    spheres = []
    initial_radius = 0.5
    for i, corner in enumerate(corners3d):
        sphere = trimesh.creation.uv_sphere(radius=initial_radius)
        sphere.visual.vertex_colors = [255, 0, 0]
        sphere.apply_translation(corner)
        spheres.append(sphere)
        initial_radius *= 0.5
    initial_radius = 0.5
    for i, corner in enumerate(corners3d_rotated):
        sphere = trimesh.creation.uv_sphere(radius=initial_radius)
        sphere.visual.vertex_colors = [0, 0, 255]
        sphere.apply_translation(corner)
        spheres.append(sphere)
        initial_radius *= 0.5

    coordinate_frame = trimesh.creation.axis()
    spheres.append(coordinate_frame)

    scene = trimesh.Scene(spheres)
    # scene.show()

    # Sort the 3D corners
    sorted_indices = np.argsort(-corners3d_rotated[:, 1])
    top_two = corners3d[sorted_indices[:2]]
    bottom_two = corners3d[sorted_indices[2:]]
    top_right, top_left = top_two if top_two[0][0] > top_two[1][0] else top_two[::-1]
    bottom_left, bottom_right = bottom_two if bottom_two[0][0] < bottom_two[1][0] else bottom_two[::-1]

    # Visualization for debugging
    top_right_sphere = trimesh.creation.uv_sphere(radius=0.1)
    top_right_sphere.apply_translation(top_right)
    top_right_sphere.visual.vertex_colors = [0, 0, 255]
    top_left_sphere = trimesh.creation.uv_sphere(radius=0.05)
    top_left_sphere.apply_translation(top_left)
    top_left_sphere.visual.vertex_colors = [0, 0, 255]
    bottom_left_sphere = trimesh.creation.uv_sphere(radius=0.05)
    bottom_left_sphere.apply_translation(bottom_left)
    bottom_left_sphere.visual.vertex_colors = [255, 255, 0]
    bottom_right_sphere = trimesh.creation.uv_sphere(radius=0.1)
    bottom_right_sphere.apply_translation(bottom_right)
    bottom_right_sphere.visual.vertex_colors = [255, 255, 0]

    scene = trimesh.Scene([top_right_sphere, top_left_sphere, bottom_left_sphere, bottom_right_sphere])
    # scene.show()

    ordered_corners_3d = np.array([top_right, top_left, bottom_left, bottom_right])

    return ordered_corners_3d


def create_arch_plane(arch_name, arch_pcd_file):
    if arch_name == "wall":
        arch_obb = create_wall_plane(arch_name, arch_pcd_file)
    elif arch_name == "floor":
        arch_obb = create_floor_plane(arch_name, arch_pcd_file)
    else:
        raise NotImplementedError
    return arch_obb


def create_wall_plane(arch_name, arch_pcd_file):
    pcd = o3d.io.read_point_cloud(arch_pcd_file)
    points = np.asarray(pcd.points)
    points_mask = np.any(points, axis=-1)
    points = points[points_mask]
    pcd.points = o3d.utility.Vector3dVector(points)

    plane_model, inliers = pcd.segment_plane(distance_threshold=0.05,
                                             ransac_n=3,
                                             num_iterations=1000)
    [a, b, c, d] = plane_model
    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([0, 1.0, 0])
    ind = np.setdiff1d(np.arange(len(points)), inliers)
    outlier_cloud = pcd.select_by_index(ind)
    outlier_cloud.paint_uniform_color([1.0, 0, 0])

    # Create a visualization of the plane
    [a, b, c, d] = plane_model
    xx, yy = np.meshgrid(np.linspace(min(points[:, 0]), max(points[:, 0]), 20),
                         np.linspace(min(points[:, 1]), max(points[:, 1]), 20))
    z = (-d - a * xx - b * yy) / c
    plane_points = np.stack([xx, yy, z], axis=-1).reshape(-1, 3)
    plane_pcd = o3d.geometry.PointCloud()
    plane_pcd.points = o3d.utility.Vector3dVector(plane_points)
    plane_pcd.paint_uniform_color([0.5, 0.5, 1])  # Light blue color for the plane

    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, plane_pcd])

    pcd = inlier_cloud
    points = np.asarray(pcd.points)
    wall_plane = ArchPlane(arch_name, plane_model, pcd=pcd, center=np.mean(points, 1))

    return wall_plane


def create_floor_plane(arch_name, arch_pcd_file):
    pcd = o3d.io.read_point_cloud(arch_pcd_file)
    points = np.asarray(pcd.points)
    points_mask = np.any(points, axis=-1)
    points = points[points_mask]
    pcd.points = o3d.utility.Vector3dVector(points)

    plane_model, inliers = pcd.segment_plane(distance_threshold=0.05,
                                             ransac_n=3,
                                             num_iterations=1000)
    # Visualize plane and pcd
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([0, 1.0, 0])
    ind = np.setdiff1d(np.arange(len(points)), inliers)
    outlier_cloud = pcd.select_by_index(ind)
    outlier_cloud.paint_uniform_color([1.0, 0, 0])

    # Create a visualization of the plane
    [a, b, c, d] = plane_model
    xx, yy = np.meshgrid(np.linspace(min(points[:, 0]), max(points[:, 0]), 20),
                         np.linspace(min(points[:, 1]), max(points[:, 1]), 20))
    z = (-d - a * xx - b * yy) / c
    plane_points = np.stack([xx, yy, z], axis=-1).reshape(-1, 3)
    plane_pcd = o3d.geometry.PointCloud()
    plane_pcd.points = o3d.utility.Vector3dVector(plane_points)
    plane_pcd.paint_uniform_color([0.5, 0.5, 1])  # Light blue color for the plane
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, plane_pcd])

    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    pcd = inlier_cloud
    points = np.asarray(pcd.points)
    floor_plane = ArchPlane(arch_name, plane_model, center=np.mean(points, 1), pcd=pcd)

    return floor_plane


def rectify_wall_plane(wall_plane, floor_plane):
    line_eq = wall_plane.compute_intersection_line(floor_plane)
    if line_eq is None:
        return wall_plane
    t = sp.symbols("t")
    p0 = line_eq.subs(t, 0)
    p1 = line_eq.subs(t, 1)

    a, b, c = sp.symbols("a b c")
    _, _, _, d = wall_plane.get_params()
    m, n, k, _ = floor_plane.get_params()

    eq1 = a*m + b*n + c*k
    eq2 = a*p0[0] + b*p0[1] + c*p0[2] + d
    eq3 = a*p1[0] + b*p1[1] + c*p1[2] + d

    solution = sp.linsolve([eq1, eq2, eq3], (a, b, c))
    if solution is sp.EmptySet:
        return wall_plane
    solution = list(solution)[0]
    if type(solution[0]) == sp.core.add.Add:
        return wall_plane
    wall_plane.set_params(a=solution[0], b=solution[1], c=solution[2])

    return wall_plane


def load_arch_plane_from_npy(npy_path):
    arch_data = np.load(npy_path, allow_pickle=True).item()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arch_data["points"])
    pcd.colors = o3d.utility.Vector3dVector(arch_data["colors"])
    pcd.normals = o3d.utility.Vector3dVector(arch_data["normals"])
    arch_plane = ArchPlane(arch_data["arch_type"], arch_data["plane_model"],
                           arch_data["center"], pcd, arch_data["arch_vertices"],
                           arch_data["arch_vertices_3d"], arch_data["arch_normal"])
    return arch_plane


def load_architecture_from_path(arch_components_path):
    gt_ids_mapping = json.load(open(f"{arch_components_path}/gt_index.json"))
    gt2idx = {}
    for idx in gt_ids_mapping:
        if not gt_ids_mapping[idx]:
            continue
        iou, gt_id = sorted(gt_ids_mapping[idx], key=lambda x: x[0], reverse=True)[0]
        gt2idx.setdefault(gt_id, []).append((iou, idx))
    
    arch_dict = {}
    for gt_id in gt2idx:
        idx = sorted(gt2idx[gt_id], key=lambda x: x[0], reverse=True)[0][1]
        arch_dict[gt_id] = load_arch_plane_from_npy(f"{arch_components_path}/arch/archplanes/wall_{idx}.npy")

    return arch_dict


class ArchPlane:
    def __init__(self, arch_type, plane_model, center=None, pcd=None,
                 arch_vertices=None, arch_vertices_3d=None, arch_normal=None) -> None:
        self.arch_type = arch_type
        self.plane_model = plane_model
        self.arch_vertices = arch_vertices
        self.arch_vertices_3d = arch_vertices_3d
        self.arch_normal = arch_normal
        self.center = center
        self.pcd = pcd
        self._set_normal()

    def _set_normal(self):
        a, b, c, _ = self.plane_model
        arch_normal = np.array([a, b, c])
        arch_normal = arch_normal / np.linalg.norm(arch_normal)
        if self.arch_type == "wall" and arch_normal[2] < 0.:
            arch_normal *= -1
        elif self.arch_type == "floor" and arch_normal[1] < 0.:
            arch_normal *= -1
        self.arch_normal = arch_normal

    def get_params(self):
        return self.plane_model

    def set_params(self, a=None, b=None, c=None, d=None):
        prev_a, prev_b, prev_c, prev_d = self.plane_model
        new_a = float(a) if a else prev_a
        new_b = float(b) if b else prev_b
        new_c = float(c) if c else prev_c
        new_d = float(d) if d else prev_d
        self.plane_model = (new_a, new_b, new_c, new_d)
        self._set_normal()
        # Recompute arch vertices
        self.instantiate_arch_element()
        self.extrude_arch_element()

    def get_y(self, x, z):
        a, b, c, d = self.plane_model
        return (-a * x - c * z - d) / b

    def get_z(self, x, y):
        a, b, c, d = self.plane_model
        return (-a * x - b * y - d) / c

    def instantiate_arch_element(self, size=1.5):
        # Create vertices of a large square in the plane
        if not self.pcd:
            if self.arch_type == "wall":
                x = y = size  # Size of the square, adjust as needed
                arch_vertices = np.array([
                    [x, y, self.get_z(x, y)],
                    [-x, y, self.get_z(-x, y)],
                    [-x, -y, self.get_z(-x, -y)],
                    [x, -y, self.get_z(x, -y)],
                ])
            else:
                x = z = size  # Size of the square, adjust as needed
                z_offset = self.center[2]
                arch_vertices = np.array([
                    [x, self.get_y(x, z_offset+z), z_offset+z],
                    [x, self.get_y(x, z_offset-z), z_offset-z],
                    [-x, self.get_y(-x, z_offset-z), z_offset-z],
                    [-x, self.get_y(-x, z_offset+z), z_offset+z],
                ])
            self.arch_vertices = arch_vertices
        else:

            arch_vertices = create_convex_hull_bounding_box(self.pcd, self)

            lines = [[i, (i+1)%len(arch_vertices)] for i in range(len(arch_vertices))]
            colors = [[1, 0, 0] for _ in range(len(lines))]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(arch_vertices)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)

            allviz.append([self.pcd, line_set])

            # o3d.visualization.draw_geometries([self.pcd, line_set])

            self.arch_vertices = arch_vertices

    def extrude_arch_element(self, extrude_distance=0.01):
        arch_vertices_extruded = self.arch_vertices + (- self.arch_normal) * extrude_distance
        self.arch_vertices_3d = np.vstack((self.arch_vertices, arch_vertices_extruded))

    def compute_intersection_line(self, other_plane):
        x, y, z, t = sp.symbols('x y z t')

        # Equations of the planes
        a, b, c, d = self.get_params()
        eq1 = a*x + b*y + c*z + d
        m, n, k, l = other_plane.get_params()
        eq2 = m*x + n*y + k*z + l

        # Cross product to find the direction vector of the intersection line
        dir_vec = np.cross(self.arch_normal, other_plane.arch_normal)

        # Solve for a point on the intersection line by setting x = 0
        solution = sp.linsolve([eq1.subs(x, 0), eq2.subs(x, 0)], (y, z))

        # Check if solution is empty (which means no intersection in the given setting)
        if solution:
            p0 = [0, *list(solution)[0]]
            line_eq = sp.Matrix(p0) + t * sp.Matrix(dir_vec)
            # print("Parametric equation of the line:")
            # print("x(t) =", line_eq[0])
            # print("y(t) =", line_eq[1])
            # print("z(t) =", line_eq[2])
        else:
            print("No intersection found.")
            line_eq = None

        return line_eq

    def compute_intersection_line_with_bounds(self, other_plane):
        x, y, z, t = sp.symbols('x y z t')

        # Equations of the planes
        a, b, c, d = self.get_params()
        eq1 = a*x + b*y + c*z + d
        m, n, k, l = other_plane.get_params()
        eq2 = m*x + n*y + k*z + l

        # Cross product to find the direction vector of the intersection line
        dir_vec = np.cross(self.arch_normal, other_plane.arch_normal)

        # Solve for a point on the intersection line by setting x = 0
        solution = sp.linsolve([eq1.subs(x, 0), eq2.subs(x, 0)], (y, z))

        if solution:
            p0 = np.array([0, *list(solution)[0]]).astype(float)
            line_eq = p0 + t * np.array(dir_vec)

            # Find t values for each coordinate at -20 and 20
            t_values = []
            for i in range(3):
                if dir_vec[i] != 0:
                    t_min = (-20 - p0[i]) / dir_vec[i]
                    t_max = (20 - p0[i]) / dir_vec[i]
                    t_values.extend([t_min, t_max])

            # Use the most extreme t values
            t_min, t_max = min(t_values), max(t_values)

            # Generate points along the line
            t_range = np.linspace(t_min, t_max, 10000)
            points = p0 + np.outer(t_range, dir_vec)

            # Filter points within bounds of either plane
            intersection_points = []
            for point in points:
                # Some approximation required here as sometimes bboxes are non-overlapping when they should be
                if self.is_point_within_bounds(point, tolerance=0.1) and other_plane.is_point_within_bounds(point, tolerance=0.1):
                    intersection_points.append(point)

            if intersection_points:
                # Visualization
                meshes = []
                for plane in [self, other_plane]:
                    mesh = plane.export_arch_mesh_3d()
                    meshes.append(mesh)

                for point in intersection_points:
                    point_sphere = trimesh.creation.uv_sphere(radius=0.1)
                    point_sphere.apply_translation(point)
                    point_sphere.visual.vertex_colors = [255, 0, 0]
                    meshes.append(point_sphere)

                scene = trimesh.Scene(meshes)
                # scene.show()

                # Return the symbolic line equation
                return sp.Matrix(p0) + t * sp.Matrix(dir_vec)
            else:
                # If current plane is the floor, return the line equation regardless
                if self.arch_type == "floor":
                    return sp.Matrix(p0) + t * sp.Matrix(dir_vec)
                print("Intersection line does not intersect with the bounds of either plane.")
                return None
        else:
            print("No intersection found.")
            return None

    def is_point_within_bounds(self, point, tolerance=0):
        x_min, y_min, z_min = np.min(self.arch_vertices, axis=0)
        x_max, y_max, z_max = np.max(self.arch_vertices, axis=0)

        return (x_min - tolerance <= point[0] <= x_max + tolerance and
                y_min - tolerance <= point[1] <= y_max + tolerance and
                z_min - tolerance <= point[2] <= z_max + tolerance)

    def project_point_to_plane(self, point):
        normal = self.arch_normal
        d = self.plane_model[3]
        t = -(np.dot(normal, point) + d) / np.dot(normal, normal)
        return point + t * normal

    def get_bounds(self):
        arch_vertices = self.arch_vertices_3d
        min_bounds = np.min(arch_vertices, axis=0)
        max_bounds = np.max(arch_vertices, axis=0)
        return (*min_bounds, *max_bounds)

    def get_surfs(self):
        surfs = {
            "support": self.arch_vertices.copy()[None, ...],
        }
        return surfs

    def get_vecs(self):
        vecs = {
            "support": self.arch_normal.copy()[None, ...],
        }
        return vecs

    def export_optimization(self):
        optim_arch = {
            "corners": self.arch_vertices_3d.copy(),
            "supp_surf": self.arch_vertices,
            "supp_vec": self.arch_normal,
        }
        return optim_arch

    def export_arch_mesh_2d(self):
        # Extend to non-specified number of vertices
        convex_vertices, new_order = infer_convex_vertices(self.arch_vertices, self.plane_model)
        faces = []
        for i in range(len(convex_vertices) - 2):
            faces.append([0, i + 1, i + 2])
        plane_mesh = trimesh.Trimesh(vertices=convex_vertices, faces=faces)
        return plane_mesh

    def export_arch_mesh_3d(self):
        convex_vertices, new_order = infer_convex_vertices(self.arch_vertices, self.plane_model)

        extruded_indices = new_order

        bottom_vertices = self.arch_vertices_3d[len(self.arch_vertices):][extruded_indices]

        new_arch_vertices_3d = np.vstack((convex_vertices, bottom_vertices))

        N = len(convex_vertices)

        faces = []

        for i in range(1, N - 1):
            faces.append([0, i, i + 1])

        for i in range(N, 2 * N - 1):
            faces.append([N, i + 1, i])

        for i in range(N):
            top_current = i
            top_next = (i + 1) % N
            bottom_current = i + N
            bottom_next = ((i + 1) % N) + N

            faces.append([top_current, top_next, bottom_next])

            faces.append([top_current, bottom_next, bottom_current])

        plane_mesh = trimesh.Trimesh(vertices=new_arch_vertices_3d, faces=faces, process=True)

        return plane_mesh

    def export_arch_mesh_3d_earcut(self, hole_vertices=None):
        # Project vertices onto plane and define 2D coordinates
        projected_points = project_to_plane(self.arch_vertices, self.plane_model)
        x_axis = np.array([1, 0, 0])
        if np.abs(np.dot(x_axis, self.arch_normal)) > 0.9:
            x_axis = np.array([0, 1, 0])
        y_axis = np.cross(self.arch_normal, x_axis)
        x_axis = np.cross(y_axis, self.arch_normal)
        x_axis /= np.linalg.norm(x_axis)
        y_axis /= np.linalg.norm(y_axis)
        center = np.mean(projected_points, axis=0)
        points_2d = np.column_stack([
            np.dot(projected_points - center, x_axis),
            np.dot(projected_points - center, y_axis)
        ])

        # Mesh the 2D points using Earcut
        vertices = points_2d.tolist()
        sorted_points, sorted_indices = sort_points_counter_clockwise(points_2d)
        vertices = [vertices[i] for i in sorted_indices]
        rings = [len(vertices)]
        if hole_vertices is not None:
            hole_points = project_to_plane(hole_vertices, self.plane_model)
            hole_points_2d = np.column_stack([
                np.dot(hole_points - center, x_axis),
                np.dot(hole_points - center, y_axis)
            ])
            vertices.extend(hole_points_2d.tolist())
            rings.append(len(hole_points_2d))
        faces = earcut.triangulate_float32(np.asarray(vertices, dtype=np.float32), np.asarray(rings, dtype=np.uint32)).reshape(-1, 3)
        new_vertices = self.arch_vertices[sorted_indices]
        if hole_vertices is not None:
            new_vertices = np.vstack((new_vertices, hole_vertices))
        plane_mesh = trimesh.Trimesh(vertices=new_vertices, faces=faces)
        return plane_mesh

    def export_obb_mesh(self):
        return self.export_arch_mesh_3d()


def infer_convex_vertices(vertices, plane_model):
    normal = plane_model[:3]
    projected_points = project_to_plane(vertices, plane_model)
    centroid = np.mean(projected_points, axis=0)

    x_axis = np.array([1, 0, 0])
    if np.abs(np.dot(x_axis, normal)) > 0.9:
        x_axis = np.array([0, 1, 0])
    y_axis = np.cross(normal, x_axis)
    x_axis = np.cross(y_axis, normal)
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)

    points_2d = np.column_stack([
        np.dot(projected_points - centroid, x_axis),
        np.dot(projected_points - centroid, y_axis)
    ])

    sorted_points, sorted_indices = sort_points_counter_clockwise(points_2d)
    if len(sorted_points) != len(vertices):
        # Visualize point order by increasing the size of the sphere
        initial_r = 0.05
        point_meshes = []
        for i, point in enumerate(vertices[sorted_indices]):
            point_sphere = trimesh.creation.uv_sphere(radius=initial_r)
            point_sphere.apply_translation(point)
            point_meshes.append(point_sphere)
            initial_r *= 1.5
        scene = trimesh.Scene(point_meshes)
        # scene.show()
    convex_point_ids = []
    num_points = len(sorted_points)

    for i in range(num_points):
        p_prev = sorted_points[i - 1]
        p_curr = sorted_points[i]
        p_next = sorted_points[(i + 1) % num_points]

        v1 = p_prev - p_curr
        v2 = p_next - p_curr

        cross_z = v1[0] * v2[1] - v1[1] * v2[0]
        if cross_z <= 0:
            convex_point_ids.append(sorted_indices[i])
    print(f"Original vertices: {len(vertices)}, convex vertices {len(convex_point_ids)}")
    if len(vertices) != len(convex_point_ids):
        for i in range(num_points):
            p_prev = sorted_points[i - 1]
            p_curr = sorted_points[i]
            p_next = sorted_points[(i + 1) % num_points]

            v1 = p_prev - p_curr
            v2 = p_next - p_curr

            cross_z = v1[0] * v2[1] - v1[1] * v2[0]
            # print(f"Cross product: {cross_z}")
        # Visualize convex vertices in gren and non-convex in red
        point_spheres = []
        for i, point in enumerate(vertices):
            point_sphere = trimesh.creation.uv_sphere(radius=0.05)
            point_sphere.apply_translation(point)
            if i in convex_point_ids:
                point_sphere.visual.vertex_colors = [0, 255, 0]
            else:
                point_sphere.visual.vertex_colors = [255, 0, 0]
            point_spheres.append(point_sphere)
        scene = trimesh.Scene(point_spheres)
        # scene.show()
    return vertices[convex_point_ids], convex_point_ids


def sort_points_counter_clockwise(points):
    center = np.mean(points, axis=0)

    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])

    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]
    return sorted_points, sorted_indices


def project_to_plane(points, plane_model):
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal_norm = np.linalg.norm(normal)
    normal = normal / normal_norm

    distances = (np.dot(points, normal) + d) / normal_norm
    projected_points = points - np.outer(distances, normal)

    return projected_points


def infer_convex_vertices(vertices, plane_model):
    normal = plane_model[:3]
    projected_points = project_to_plane(vertices, plane_model)
    centroid = np.mean(projected_points, axis=0)

    x_axis = np.array([1, 0, 0])
    if np.abs(np.dot(x_axis, normal)) > 0.9:
        x_axis = np.array([0, 1, 0])
    y_axis = np.cross(normal, x_axis)
    x_axis = np.cross(y_axis, normal)
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)

    points_2d = np.column_stack([
        np.dot(projected_points - centroid, x_axis),
        np.dot(projected_points - centroid, y_axis)
    ])

    sorted_points, sorted_indices = sort_points_counter_clockwise(points_2d)
    if len(sorted_points) != len(vertices):
        # Visualize point order by increasing the size of the sphere
        initial_r = 0.05
        point_meshes = []
        for i, point in enumerate(vertices[sorted_indices]):
            point_sphere = trimesh.creation.uv_sphere(radius=initial_r)
            point_sphere.apply_translation(point)
            point_meshes.append(point_sphere)
            initial_r *= 1.5
        scene = trimesh.Scene(point_meshes)
        # scene.show()
    convex_point_ids = []
    num_points = len(sorted_points)

    for i in range(num_points):
        p_prev = sorted_points[i - 1]
        p_curr = sorted_points[i]
        p_next = sorted_points[(i + 1) % num_points]

        v1 = p_prev - p_curr
        v2 = p_next - p_curr

        cross_z = v1[0] * v2[1] - v1[1] * v2[0]
        if cross_z <= 0:
            convex_point_ids.append(sorted_indices[i])
    # print(f"Original vertices: {len(vertices)}, convex vertices {len(convex_point_ids)}")
    if len(vertices) != len(convex_point_ids):
        for i in range(num_points):
            p_prev = sorted_points[i - 1]
            p_curr = sorted_points[i]
            p_next = sorted_points[(i + 1) % num_points]

            v1 = p_prev - p_curr
            v2 = p_next - p_curr

            cross_z = v1[0] * v2[1] - v1[1] * v2[0]
            # print(f"Cross product: {cross_z}")
        # Visualize convex vertices in gren and non-convex in red
        point_spheres = []
        for i, point in enumerate(vertices):
            point_sphere = trimesh.creation.uv_sphere(radius=0.05)
            point_sphere.apply_translation(point)
            if i in convex_point_ids:
                point_sphere.visual.vertex_colors = [0, 255, 0]
            else:
                point_sphere.visual.vertex_colors = [255, 0, 0]
            point_spheres.append(point_sphere)
        # scene = trimesh.Scene(point_spheres)
        # scene.show()
    return vertices[convex_point_ids], convex_point_ids


def project_to_plane(points, plane_model):
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal_norm = np.linalg.norm(normal)
    normal = normal / normal_norm

    distances = (np.dot(points, normal) + d) / normal_norm
    projected_points = points - np.outer(distances, normal)

    return projected_points


def planes_intersect(planes):
    intersection_dict = {}
    for i, plane in enumerate(planes):
        for j, plane2 in enumerate(planes):
            if i == j:
                continue
            # Check if planes intersect
            intersection_equation = plane.compute_intersection_line_with_bounds(plane2)
            if intersection_equation is None:
                continue
            else:
                if i not in intersection_dict:
                    intersection_dict[i] = {}
                intersection_dict[i][j] = intersection_equation
    return intersection_dict


def project_vertices_onto_line(vertices, line_eq):
    t = sp.symbols('t')

    direction = np.array([float(line_eq[i].diff(t)) for i in range(3)])
    line_point = np.array([float(line_eq[i].subs(t, 0)) for i in range(3)])

    direction = direction / np.linalg.norm(direction)

    projected_points = []

    for vertex in vertices:
        v = vertex - line_point
        t_proj = np.dot(v, direction)
        projected_point = line_point + t_proj * direction
        projected_points.append(projected_point)
    return np.array(projected_points)


def infer_bounds(intersection_dict, all_planes, floor_idx):
    adjusted_track = {}
    for i, intersects in intersection_dict.items():
        # print(f"Processing plane {i} with intersections: {intersects}")
        if i not in adjusted_track:
            adjusted_track[i] = [False] * len(all_planes[i].arch_vertices)
        if i == (floor_idx - 1):
            continue
        wall_plane = all_planes[i]

        # Visualize vertices in order by increasing the size of the sphere
        initial_r = 0.1
        point_meshes = []
        for point in wall_plane.arch_vertices:
            point_sphere = trimesh.creation.uv_sphere(radius=initial_r)
            point_sphere.apply_translation(point)
            point_meshes.append(point_sphere)
            initial_r *= 1.5
        scene = trimesh.Scene(point_meshes)
        # scene.show()

        for j, intersection_eq in intersects.items():
            # print(f"\tProcessing intersection with plane {j}")
            if j not in adjusted_track:
                adjusted_track[j] = [False] * len(all_planes[j].arch_vertices)
            if j == (floor_idx - 1):
                continue
            # Find vertices closest to the intersection line for both planes
            plane1 = all_planes[i]
            plane2 = all_planes[j]
            projected_vertices = project_vertices_onto_line(plane1.arch_vertices + 1e-15, intersection_eq)
            projected_distances = np.linalg.norm(projected_vertices - plane1.arch_vertices, axis=1)
            if j == (floor_idx - 1):
                # Check that we select vertices below the floor plane if possible
                closest_ids = np.argsort(projected_distances)
                selected_ids = []
                for idx in closest_ids:
                    vertex = plane1.arch_vertices[idx]
                    y_plane = plane2.get_y(vertex[0], vertex[2])
                    if vertex[1] < y_plane:
                        selected_ids.append(idx)
                if len(selected_ids) < 2:
                    selected_ids += [idx for idx in closest_ids if idx not in selected_ids]
                closest_ids = selected_ids[:2]
            else:
                closest_ids = np.argsort(projected_distances)[:2]

            disatnces20 = np.linalg.norm(plane2.arch_vertices - projected_vertices[closest_ids[0]], axis=1)
            temp_ids = np.argsort(disatnces20)
            closest_id20 = None
            for temp_id in temp_ids:
                # Should be closest and same side (with respect to y)
                if closest_ids[0] in [0, 1] and temp_id in [0, 1]:
                    closest_id20 = temp_id
                    break
                elif closest_ids[0] in [2, 3] and temp_id in [2, 3]:
                    closest_id20 = temp_id
                    break
            disatnces21 = np.linalg.norm(plane2.arch_vertices - projected_vertices[closest_ids[1]], axis=1)
            temp_ids = np.argsort(disatnces21)
            closest_id21 = None
            for temp_id in temp_ids:
                # Should be closest and same side (with respect to y)
                if closest_ids[1] in [0, 1] and temp_id in [0, 1]:
                    closest_id21 = temp_id
                    break
                elif closest_ids[1] in [2, 3] and temp_id in [2, 3]:
                    closest_id21 = temp_id
                    break
            closest_ids2 = [closest_id20, closest_id21]

            # Visualize
            original_vertex = trimesh.creation.uv_sphere(radius=0.1)
            original_vertex.apply_translation(plane1.arch_vertices[closest_ids[0]])
            original_vertex.visual.vertex_colors = [255, 0, 0]
            original_vertex2 = trimesh.creation.uv_sphere(radius=0.1)
            original_vertex2.apply_translation(plane1.arch_vertices[closest_ids[1]])
            original_vertex2.visual.vertex_colors = [255, 0, 0]
            projected_vertex = trimesh.creation.uv_sphere(radius=0.1)
            projected_vertex.apply_translation(projected_vertices[closest_ids[0]])
            projected_vertex.visual.vertex_colors = [0, 255, 0]
            projected_vertex2 = trimesh.creation.uv_sphere(radius=0.1)
            projected_vertex2.apply_translation(projected_vertices[closest_ids[1]])
            projected_vertex2.visual.vertex_colors = [0, 255, 0]
            # viz_planes([plane1, plane2, original_vertex, original_vertex2, projected_vertex, projected_vertex2])

            if (floor_idx - 1) in intersects and (floor_idx - 1) in intersection_dict[j] and (floor_idx - 1) != j:
                floor_eq = intersects[floor_idx - 1]
                # Visualize planes and intersection line
                # Sample points along the intersection line
                t = sp.symbols('t')
                line_min = floor_eq.subs(t, -20)
                line_max = floor_eq.subs(t, 20)

                lineset = o3d.geometry.LineSet()
                lineset.points = o3d.utility.Vector3dVector([line_min, line_max])
                lineset.lines = o3d.utility.Vector2iVector([[0, 1]])
                lineset.colors = o3d.utility.Vector3dVector([[0, 0, 1], [0, 0, 1]])

                # o3d.visualization.draw_geometries([plane1.pcd, plane2.pcd, lineset])

                projected_vertices_floor = project_vertices_onto_line(projected_vertices + 1e-15, floor_eq)[closest_ids]
                projected_distances = np.linalg.norm(projected_vertices_floor - projected_vertices[closest_ids], axis=1)
                closest_idx_floor = np.argsort(projected_distances)
                closest_id_floor = None
                for idx in closest_idx_floor:
                    vertex = projected_vertices[closest_ids[idx]]
                    floor_plane = all_planes[floor_idx - 1]
                    y_floor = floor_plane.get_y(vertex[0], vertex[2])
                    if vertex[1] <= y_floor:
                        closest_id_floor = idx
                        break
                if closest_id_floor is None:
                    # Pick the closest one
                    closest_id_floor = closest_idx_floor[0]
                other_id = 1 - closest_id_floor
                # Visualize
                original_vertex = trimesh.creation.uv_sphere(radius=0.1)
                original_vertex.apply_translation(projected_vertices[closest_ids[0]])
                original_vertex.visual.vertex_colors = [255, 0, 0]
                original_vertex2 = trimesh.creation.uv_sphere(radius=0.1)
                original_vertex2.apply_translation(projected_vertices[closest_ids[1]])
                original_vertex2.visual.vertex_colors = [255, 0, 0]
                projected_vertex = trimesh.creation.uv_sphere(radius=0.1)
                projected_vertex.apply_translation(projected_vertices_floor[0])
                projected_vertex.visual.vertex_colors = [0, 255, 0]
                projected_vertex2 = trimesh.creation.uv_sphere(radius=0.1)
                projected_vertex2.apply_translation(projected_vertices_floor[1])
                projected_vertex2.visual.vertex_colors = [0, 255, 0]
                # viz_planes([plane1, plane2, all_planes[floor_idx - 1], original_vertex, original_vertex2, projected_vertex, projected_vertex2])

                # This is trated as the "corner" - intersection of two planes with the floor
                corner = projected_vertices_floor[closest_id_floor]

                # Two vertices from the same plane should be adjusted at a time - select a plane to adjust
                if not adjusted_track[i][closest_ids[closest_id_floor]] and not adjusted_track[i][closest_ids[other_id]]:
                    plane1.arch_vertices[closest_ids[closest_id_floor]] = projected_vertices[closest_ids[closest_id_floor]]
                    plane1.arch_vertices[closest_ids[other_id]] = projected_vertices[closest_ids[other_id]]
                    adjusted_track[i][closest_ids[closest_id_floor]] = True
                    adjusted_track[j][closest_ids[other_id]] = True
                elif not adjusted_track[j][closest_ids2[other_id]] and not adjusted_track[j][closest_ids2[other_id]]:
                    plane2.arch_vertices[closest_ids2[closest_id_floor]] = projected_vertices[closest_ids[closest_id_floor]]
                    plane2.arch_vertices[closest_ids2[other_id]] = projected_vertices[closest_ids[other_id]]
                    adjusted_track[j][closest_ids2[closest_id_floor]] = True
                    adjusted_track[j][closest_ids2[other_id]] = True

                # DEBUG visualize after adjustment
                # viz_planes([plane1, plane2, all_planes[floor_idx - 1]])

            else:
                # If no floor intersection, just updated two closest vertices without additional projection on the floor
                # print(f"Planes {i} and {j} do not intersect with the floor.")

                # DEBUG visualize before adjustment
                # viz_planes([plane1, plane2])

                # Update two vertices of 1 plane at a time
                if not adjusted_track[i][closest_ids[0]] and not adjusted_track[i][closest_ids[1]]:
                    plane1.arch_vertices[closest_ids[0]] = projected_vertices[closest_ids[0]]
                    plane1.arch_vertices[closest_ids[1]] = projected_vertices[closest_ids[1]]
                    adjusted_track[i][closest_ids[0]] = True
                    adjusted_track[i][closest_ids[1]] = True
                elif not adjusted_track[j][closest_ids2[0]] and not adjusted_track[j][closest_ids2[1]]:
                    plane2.arch_vertices[closest_ids2[0]] = projected_vertices[closest_ids[0]]
                    plane2.arch_vertices[closest_ids2[1]] = projected_vertices[closest_ids[1]]
                    adjusted_track[j][closest_ids2[0]] = True
                    adjusted_track[j][closest_ids2[1]] = True
                # DEBUG visualize after adjustment
                # viz_planes([plane1, plane2])

    # Infer floor plane bounds
    # Conduct tests to determine invisible vertices
    # Add all the wall vertices that are connected to the floor plane, to the floor plane vertices
    # Drop all invisble vertices and mesh the floor plane
    visible_vertex_check = {0: -1, 1: -1, 2: -1, 3: -1}
    if floor_idx >= 0:
        floor_plane = all_planes[floor_idx - 1]
        for i, floor_vertex in enumerate(all_planes[floor_idx - 1].arch_vertices):
            for wall_plane in all_planes[:floor_idx - 1] + all_planes[floor_idx:]:
                # Conduct intersection test

                # Define the line from origin to floor vertex
                origin = np.array([0, 0, 0])
                direction = np.array(floor_vertex) - origin

                # Get plane parameters
                a, b, c, d = wall_plane.get_params()
                plane_normal = np.array([a, b, c])

                # Check if line and plane are parallel
                if np.dot(plane_normal, direction) == 0:
                    # Line is parallel to plane, no intersection
                    continue

                # Calculate intersection
                t = -d / np.dot(plane_normal, direction)
                intersection_point = origin + t * direction

                if t > 0 and t < 1:  # Intersection occurs between origin and floor vertex
                    # Visualize
                    point_sphere = trimesh.creation.uv_sphere(radius=0.1)
                    point_sphere.apply_translation(intersection_point)
                    point_sphere.visual.vertex_colors = [255, 0, 0]

                    floor_vertex_sphere = trimesh.creation.uv_sphere(radius=0.1)
                    floor_vertex_sphere.apply_translation(floor_vertex)
                    floor_vertex_sphere.visual.vertex_colors = [0, 255, 0]

                    # viz_planes([wall_plane, point_sphere, floor_vertex_sphere])

                    # print(f"Point {i} within bounds?: {wall_plane.is_point_within_bounds(intersection_point)}")
                    if not wall_plane.is_point_within_bounds(intersection_point):
                        if visible_vertex_check[i] != 0:
                            visible_vertex_check[i] = 1
                    else:
                        visible_vertex_check[i] = 0
                else:
                    # No intersection between origin and floor vertex
                    if visible_vertex_check[i] != 0:
                        visible_vertex_check[i] = 1

                # print(f"Visible vertex check: {visible_vertex_check}")
        visible_vertices = []
        for i, check in visible_vertex_check.items():
            if check == 1:
                visible_vertices.append(all_planes[floor_idx - 1].arch_vertices[i])
        # Now, select the vertices from each wall plane that are adjacent to the floor plane
        # viz_planes(all_planes)
        if len(all_planes) > 1:
            for i, wall_plane in enumerate(all_planes[:floor_idx - 1] + all_planes[floor_idx:]):
                # Select the two with min y, if sufficiently close to the floor plane
                candidates = wall_plane.arch_vertices[2:]
                for candidate in candidates:
                    candidate_floor_projected = project_to_plane([candidate], all_planes[floor_idx - 1].plane_model)[0]
                    if np.linalg.norm(candidate_floor_projected - candidate) < 0.1:
                        # Check if the candidate is enclosed by already added vertices
                        # if not floor_plane.is_point_within_bounds(candidate_floor_projected):
                        visible_vertices.append(candidate)

                # Visualize vertices in order by increasing the size of the sphere
                initial_r = 0.1
                point_meshes = []
                for point in wall_plane.arch_vertices:
                    point_sphere = trimesh.creation.uv_sphere(radius=initial_r)
                    point_sphere.apply_translation(point)
                    point_meshes.append(point_sphere)
                    initial_r *= 1.5
                scene = trimesh.Scene(point_meshes)
                # scene.show()

            all_planes[floor_idx - 1].arch_vertices = np.array(visible_vertices)
            all_planes[floor_idx - 1].extrude_arch_element()
            visible_vertices_spheres = []
            for vertex in visible_vertices:
                sphere = trimesh.creation.uv_sphere(radius=0.1)
                sphere.apply_translation(vertex)
                sphere.visual.vertex_colors = [255, 0, 0]
                visible_vertices_spheres.append(sphere)
            # viz_planes(all_planes + visible_vertices_spheres)
    return all_planes


def farthest_point_downsampling_idx_pointops(points, n_samples):
    import pointops
    
    points_tensor = torch.from_numpy(points).float().cuda()

    N = points.shape[0]
    offset = torch.tensor([N], device='cuda')

    new_offset = torch.tensor([n_samples], device='cuda')

    sampled_indices = pointops.farthest_point_sampling(points_tensor, offset, new_offset)

    sampled_indices_np = sampled_indices.cpu().numpy()

    return sampled_indices_np

def sample_points(arch_mesh, num_points=20000):
    points = trimesh.sample.sample_surface(arch_mesh, count=800000)[0]
    sampled_indices = farthest_point_downsampling_idx_pointops(points, num_points)
    points = points[sampled_indices]
    return points


def viz_planes(planes):
    meshes = []
    for plane in planes:
        if isinstance(plane, ArchPlane):
            mesh = plane.export_arch_mesh_2d()
            meshes.append(mesh)
        elif isinstance(plane, trimesh.Trimesh):
            meshes.append(plane)
    scene = trimesh.Scene(meshes)
    scene.show()


if __name__ == '__main__':
    segmentation_type = "segmentation_input_name"
    export_type = "output_name"
    exp_paths = ["path/to/experiment"]

    import time
    start = time.time()
    for exp_path in exp_paths:
        scene_dirs = glob(f"{exp_path}/scene*")
        for scene_dir in tqdm(scene_dirs):
            scene_id = scene_dir.split("/")[-1]
            allviz = []
            print(f"Processing scene {scene_id} ({scene_dir})")

            # Initialization
            floor_plane = None
            floor_mesh = None
            with open(f"{scene_dir}/{segmentation_type}/arch/semantic_reference.json", "r") as f:
                sem_ref = json.load(f)
            if len(sem_ref) == 0:
                continue
            for idx, label in sem_ref.items():
                if label == "floor":
                    floor_plane = create_floor_plane("floor", f"{scene_dir}/{segmentation_type}/clusters/wall_{idx}.ply")
                    floor_plane.instantiate_arch_element()
                    floor_plane.extrude_arch_element()
                    floor_mesh = floor_plane.export_arch_mesh_3d()
                    floor_idx = int(idx)

            wall_planes = []
            wall_meshes = []
            pre_rectify_wall_meshes = []

            if os.path.exists(f"{scene_dir}/{export_type}/arch/components"):
                os.system(f"rm -r {scene_dir}/{export_type}/arch/components")
            os.makedirs(f"{scene_dir}/{export_type}/arch/components", exist_ok=True)
            os.makedirs(f"{scene_dir}/{export_type}/arch/archplanes", exist_ok=True)
            index_map = {}
            index = 0
            for idx, label in sem_ref.items():
                if label == "wall":
                    wall_path = f"{scene_dir}/{segmentation_type}/clusters/wall_{idx}.ply"
                else:
                    continue
                wall_plane = create_wall_plane("wall", wall_path)
                wall_plane.instantiate_arch_element()
                if floor_plane:
                    wall_plane = rectify_wall_plane(wall_plane, floor_plane)
                wall_plane.extrude_arch_element()
                wall_mesh = wall_plane.export_arch_mesh_3d()
                wall_meshes.append(wall_mesh)
                wall_planes.append(wall_plane)
                index_map[index] = idx
                index += 1

            # Originally, bounds are not computed properly therefore need to post-process
            # Infer bounds
            if floor_plane:
                all_planes = wall_planes[:floor_idx - 1] + [floor_plane] + wall_planes[floor_idx - 1:]
            else:
                all_planes = wall_planes
                floor_idx = -1
            # Make sure all sufficiently close vertices to the floor are actually projected onto the floor first
            # Then, infer bounds
            if floor_plane is not None:
                for wall_plane in wall_planes:
                    projected_vertices = project_to_plane(wall_plane.arch_vertices, floor_plane.plane_model)
                    norms = np.linalg.norm(projected_vertices - wall_plane.arch_vertices, axis=1)
                    print(f"Norms: {norms}")
                    sorted_indices = np.argsort(norms)[:2]  # Maximum of 2 points are projected onto the floor
                    for idx in sorted_indices:
                        if norms[idx] < 0.3:
                            wall_plane.arch_vertices[idx] = projected_vertices[idx]

            scene = trimesh.Scene(wall_meshes + [floor_mesh] if floor_mesh else wall_meshes)
            # scene.show()
            intersection_dict = planes_intersect(all_planes)
            all_planes = infer_bounds(intersection_dict, all_planes, floor_idx)
            wall_planes = all_planes[:floor_idx - 1] + all_planes[floor_idx:]
            floor_mesh = None
            if floor_idx != -1:
                floor_plane = all_planes[floor_idx - 1]
                floor_mesh = floor_plane.export_arch_mesh_3d()
            wall_meshes = []
            for plane in wall_planes:
                plane.extrude_arch_element()
                wall_mesh = plane.export_arch_mesh_3d()
                wall_meshes.append(wall_mesh)

            # Exporting
            if floor_mesh:
                arch_mesh = floor_mesh + sum(wall_meshes)
            else:
                arch_mesh = sum(wall_meshes)
            # o3d.visualization.draw_geometries([viz_comp for viz in allviz for viz_comp in viz])
            arch_mesh.export(f"{scene_dir}/{export_type}/arch/arch.ply")
            arch_points = sample_points(arch_mesh)
            np.savez(f"{scene_dir}/{export_type}/arch/arch_points.npz", points=arch_points)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(arch_points)
            o3d.io.write_point_cloud(f"{scene_dir}/{export_type}/arch/arch_points.ply", pcd)
            if floor_mesh:
                floor_mesh.export(f"{scene_dir}/{export_type}/arch/components/wall_{floor_idx}.ply")
                floor_dict = {
                    "arch_type": "floor",
                    "plane_model": floor_plane.get_params(),
                    "center": floor_plane.center,
                    "arch_vertices": floor_plane.arch_vertices,
                    "arch_vertices_3d": floor_plane.arch_vertices_3d,
                    "arch_normal": floor_plane.arch_normal,
                    "points": np.asarray(floor_plane.pcd.points),
                    "colors": np.asarray(floor_plane.pcd.colors),
                    "normals": np.asarray(floor_plane.pcd.normals)
                }
                np.save(f"{scene_dir}/{export_type}/arch/archplanes/wall_{floor_idx}.npy", floor_dict)
                np.save(f"{scene_dir}/{export_type}/arch/components/wall_{floor_idx}.npy", floor_dict["plane_model"])
            for i, wall_plane in enumerate(wall_planes):
                wall_meshes[i].export(f"{scene_dir}/{export_type}/arch/components/wall_{index_map[i]}.ply")
                wall_dict = {
                    "arch_type": "wall",
                    "plane_model": wall_plane.get_params(),
                    "center": wall_plane.center,
                    "arch_vertices": wall_plane.arch_vertices,
                    "arch_vertices_3d": wall_plane.arch_vertices_3d,
                    "arch_normal": wall_plane.arch_normal,
                    "points": np.asarray(wall_plane.pcd.points),
                    "colors": np.asarray(wall_plane.pcd.colors),
                    "normals": np.asarray(wall_plane.pcd.normals)
                }
                np.save(f"{scene_dir}/{export_type}/arch/archplanes/wall_{index_map[i]}.npy", wall_dict)
                np.save(f"{scene_dir}/{export_type}/arch/components/wall_{index_map[i]}.npy",
                        wall_dict["plane_model"])
            print(f"Processed scene {scene_id} ({scene_dir})")
    print(f"Elapsed time: {time.time() - start}")