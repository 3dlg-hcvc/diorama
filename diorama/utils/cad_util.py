import os
import json
import numpy as np
from PIL import Image
import trimesh
from scipy.spatial.transform import Rotation


CAD_DIR = {
    "HSSD": "/datasets/internal/hssd/fp-models/glb-noq/{subdir}/{cad_id}.glb",
    "Objaverse": "/datasets/external/objaverse/glbs/{subdir}/{cad_id}.glb",
    "S2C": "/datasets/internal/ShapeNetCore.v2/{subdir}/{cad_id}/models/model_normalized.obj",
    "WSS": "data/wss/databaseFull/models/{cad_id}.obj"
}

RENDER_DIR = {
    'HSSD': 'data/fpmodels/fpmodels-neutral-renders-{n_ref}/{subdir}/{cad_id}',
    'Objaverse': 'data/objaverse/objaverse-lvis-neutral-renders-{n_ref}/{subdir}/{cad_id}',
    'S2C': 'data/scan2cad/shapenet-neutral-renders-{n_ref}/{cad_id}',
    "WSS": "data/wss-neutral-renders-{n_ref}/{cad_id}"
}


def parse_cad_entry(cad_entry):
    cad_info_list = cad_entry.split(",")
    if len(cad_info_list) == 4:
        idx, dataset, subdir, cad_id = cad_info_list
        up, front = None, None
    else:
        idx, dataset, subdir, cad_id, up, front = cad_info_list
        up = [float(x) for x in up.split(' ')] if up else None
        front = [float(x) for x in front.split(' ')] if front else None
    
    return idx, dataset, subdir, cad_id, up, front


def load_mesh(cad_entry_or_obj_path: str, normalize=True, load_textures=False, base_color=[150, 150, 150, 255], up=None, front=None, is_OG=False, reorient_mat=None):
    if os.path.isfile(cad_entry_or_obj_path):
        obj_path = cad_entry_or_obj_path
    else:
        _, dataset, subdir, cad_id, up, front = parse_cad_entry(cad_entry_or_obj_path)
        obj_path = CAD_DIR[dataset].format(subdir=subdir, cad_id=cad_id)
    
    if load_textures:
        mesh = trimesh.load(obj_path)
        if obj_path.endswith('.glb'):
            for _, sub_mesh in mesh.geometry.items():
                if sub_mesh.visual.material.baseColorTexture is None:
                    if sub_mesh.visual.material.baseColorFactor is None:
                        baseColorFactor = np.array([255, 255, 255, 255], dtype=np.uint8)
                        sub_mesh.visual.material.baseColorFactor = np.array([255, 255, 255, 255], dtype=np.uint8)
                    else:
                        baseColorFactor = sub_mesh.visual.material.baseColorFactor
                    sub_mesh.visual.material.baseColorTexture = Image.fromarray(np.tile(baseColorFactor, 4).reshape((2, 2, 4)).astype(np.uint8))
                if sub_mesh.visual.uv is None:
                    sub_mesh.visual = trimesh.visual.ColorVisuals(sub_mesh, vertex_colors=np.repeat(baseColorFactor[None,:], len(sub_mesh.vertices), axis=0))
    else:
        mesh = trimesh.load(obj_path, force="mesh")
        baseColorFactor = np.array(base_color, dtype=np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=np.repeat(baseColorFactor[None,:], len(mesh.vertices), axis=0))
    
    if normalize:
        center = np.mean(mesh.bounds, axis=0)
        scale = 1. / max(mesh.bounds[1] - mesh.bounds[0])
        scale *= 0.85
        center_mat = np.array([
        [1, 0, 0, -center[0]],
        [0.0, 1, 0.0, -center[1]],
        [0.0, 0.0, 1, -center[2]],
        [0.0,  0.0, 0.0, 1.0]
        ])
        scale_mat = np.array([
            [scale, 0, 0, 0],
            [0.0, scale, 0.0, 0],
            [0.0, 0.0, scale, 0],
            [0.0,  0.0, 0.0, 1.0]
        ])
        rot_mat = np.eye(4)
        if up is not None and front is not None:
            rot_x = np.array([[1,0,0], up, front])
            rot_mat[:3, :3] = rot_x
        norm_mat = rot_mat @ scale_mat @ center_mat
        mesh.apply_transform(norm_mat)
        return mesh, norm_mat
    
    return mesh


def load_mesh_and_save_obb(cad_entry, normalize=True, load_textures=False, save_dir=False, obj_id=None):
    from diorama.utils.obb_util import OBB
    mesh, norm_mat = load_mesh(cad_entry, normalize=normalize, load_textures=load_textures)
    
    obb = OBB.create_from_trimesh(mesh, cad_entry, norm_mat=norm_mat)
    
    if save_dir:
        assert obj_id is not None
        os.makedirs(save_dir, exist_ok=True)
        obb.export_json(os.path.join(save_dir, f'{obj_id}.obb.json'))
        # obb_mesh = obb.export_obb_mesh_w_surfs()
        # obb_mesh.export(os.path.join(save_dir, f'{obj_id}.obb.ply'))
        # mesh.export(os.path.join(save_dir, f'{obj_id}.ply'))


def parse_glb_textures(mesh):
    for node in mesh.graph.nodes_geometry:
        pose, geom_name = mesh.graph[node]
        mesh.geometry[geom_name].apply_transform(pose)
    
    verts = []
    faces = []
    colors = []
    num_verts = 0
    for _, sub_mesh in mesh.geometry.items():
        verts.append(np.asarray(sub_mesh.vertices))
        faces.append(np.asarray(sub_mesh.faces) + num_verts)
        num_verts += len(verts[-1])
        if sub_mesh.visual.material.baseColorTexture is None:
            if sub_mesh.visual.material.baseColorFactor is None:
                sub_mesh.visual.material.baseColorFactor = np.array([255, 255, 255, 255], dtype=np.uint8)
            sub_mesh.visual.material.baseColorTexture = Image.fromarray(np.tile(sub_mesh.visual.material.baseColorFactor, 4).reshape((2, 2, 4)).astype(np.uint8))
        if sub_mesh.visual.uv is None:
            if sub_mesh.visual.material.baseColorFactor is None:
                sub_mesh.visual.material.baseColorFactor = np.array([255, 255, 255, 255], dtype=np.uint8)
            colors.append(np.repeat(sub_mesh.visual.material.baseColorFactor[None,:], len(sub_mesh.vertices), axis=0))
        else:
            sub_mesh.visual.uv = np.mod(sub_mesh.visual.uv, 1) # uv wrapping repeat in [0, 1]
            colors.append(np.asarray(sub_mesh.visual.to_color().vertex_colors))
    verts = np.concatenate(verts, axis=0)
    faces = np.concatenate(faces, axis=0)
    colors = np.concatenate(colors, axis=0)
    return verts, faces, colors