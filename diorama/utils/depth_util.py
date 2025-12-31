import os

import numpy as np
from plyfile import PlyData, PlyElement


DEPTH_SCALE = 1000
INTRINSICS = {
    "nyu": (518.8579, 519.4696, 325.58246, 253.73616, 480, 640), # (fx, fy, cx, cy, height, width)
    "multiview": (224/(2*np.tan((np.pi/3)/2)), 224/(2*np.tan((np.pi/3)/2)), 112, 112, 224, 224),
    "multiview-378": (378/(2*np.tan((np.pi/3)/2)), 378/(2*np.tan((np.pi/3)/2)), 189, 189, 378, 378),
    "wss": (784/(2*np.tan((np.pi/3)/2)), 784/(2*np.tan((np.pi/3)/2)), 504, 392, 784, 1008)
}


def load_K(data_source=None, scene_name=None):
    if data_source == 's2c':
        assert scene_name is not None, "scene_name must be provided for s2c intrinsics"
        K = load_s2c_K(f"data/scan2cad/scenes/{scene_name}")[:3, :3]
    elif data_source in INTRINSICS:
        K = make_K_from_intrinsics(INTRINSICS[data_source])
    else:
        K = make_K_from_intrinsics(INTRINSICS["wss"])
    return K


def make_K_from_intrinsics(intrinsics):
    f_x, f_y, c_x, c_y = intrinsics[:4]
    return np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])


def load_s2c_K(scene_dir: str):
    path = os.path.join(scene_dir, 'cam_intrinsics.txt')
    with open(path) as f:
        intrinsics = np.array([
            [float(e) for e in line.strip().split()]
            for line in f
        ])
    return intrinsics


def get_yfov_from_K(K, image_height):
    # Extract focal length in the y-direction (fy) from the intrinsics matrix
    fy = K[1, 1]
    
    # Compute the vertical field of view in radians
    yfov_rad = 2 * np.arctan(image_height / (2 * fy))
    
    return yfov_rad


def back_project_depth_to_points(depth_map, intrinsics="wss", image=None, normal_map=None, save_path=None, alpha_mask=None, return_pixel_map=False):
    depth_h, depth_w = depth_map.shape[:2]
    
    if isinstance(intrinsics, str):
        f_x, f_y, c_x, c_y, H, W = INTRINSICS[intrinsics]
        f_x *= depth_w / W
        f_y *= depth_h / H
        c_x *= depth_w / W
        c_y *= depth_h / H
    else:
        assert isinstance(intrinsics, np.ndarray)
        f_x, f_y = intrinsics[0, 0], intrinsics[1, 1]
        c_x, c_y = intrinsics[0, 2], intrinsics[1, 2]

    x, y = np.meshgrid(np.arange(depth_w), np.arange(depth_h))
    z = np.array(depth_map)
    x = (x - c_x) * z / f_x
    y = (y - c_y) * z / f_y
    points = np.stack((x, y, z), axis=-1)
    points[:, :, 1:] *= -1

    pixel_coords = np.stack(np.meshgrid(np.arange(depth_w), np.arange(depth_h)), axis=-1)

    if normal_map is not None:
        normals = np.array(normal_map)
        normals = normals.reshape(depth_h, depth_w, 3)
        normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
        normals[:, :, 1:] *= -1  # Flip Y and Z components to match point coordinate system

    if save_path or alpha_mask is not None:
        # Reshape points to 2D array
        points_2d = points.reshape(-1, 3)
        pixel_coords_2d = pixel_coords.reshape(-1, 2)

        # Prepare data for PLY file
        data = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        
        if image is not None:
            assert image.shape[:2] == depth_map.shape[:2], "Image shape must match depth map shape"
            colors = np.array(image).reshape(-1, 3)
            data.extend([('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        
        if normal_map is not None:
            normals_2d = normals.reshape(-1, 3)
            data.extend([('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])

        # Apply alpha mask if provided
        if alpha_mask is not None:
            assert alpha_mask.shape == depth_map.shape, "Alpha mask shape must match depth map shape"
            mask = alpha_mask.reshape(-1) > 0
            points_2d = points_2d[mask]
            pixel_coords_2d = pixel_coords_2d[mask]
            if image is not None:
                colors = colors[mask]
            if normal_map is not None:
                normals_2d = normals_2d[mask]

        if save_path:
            # Create structured array
            vertex_data = np.empty(points_2d.shape[0], dtype=data)
            vertex_data['x'], vertex_data['y'], vertex_data['z'] = points_2d.T

            if image is not None:
                vertex_data['red'], vertex_data['green'], vertex_data['blue'] = colors.T

            if normal_map is not None:
                vertex_data['nx'], vertex_data['ny'], vertex_data['nz'] = normals_2d.T

            # Create PlyElement and save to file
            vertex_element = PlyElement.describe(vertex_data, 'vertex')
            PlyData([vertex_element]).write(save_path)

        # Calculate point_to_pixel mapping after masking
        if return_pixel_map:
            point_to_pixel = {i: tuple(coord) for i, coord in enumerate(pixel_coords_2d)}
        else:
            return points
    else:
        if return_pixel_map:
            # If no alpha mask and no save_path, calculate point_to_pixel for all points
            point_to_pixel = {i: tuple(coord) for i, (point, coord) in enumerate(zip(points.reshape(-1, 3), pixel_coords.reshape(-1, 2)))}
        else:
            return points
    return points, point_to_pixel
