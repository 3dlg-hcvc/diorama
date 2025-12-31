import os
import json
import numpy as np
from PIL import Image
import colorsys

os.environ["PYOPENGL_PLATFORM"] = "egl" #opengl seems to only work with TPU

import trimesh
import pyrender
from pyrender import RenderFlags
from pyrender import PerspectiveCamera,\
                     DirectionalLight, \
                     SpotLight, \
                     OffscreenRenderer

from diorama.utils.cad_util import load_mesh
from diorama.utils.depth_util import load_K, get_yfov_from_K


# colors: [
#       '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78',
#       '#2ca02c', '#98df8a', '#d62728', '#ff9896',
#       '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
#       '#e377c2', '#f7b6d2', '#bcbd22', '#dbdb8d',
#       '#17becf', '#9edae5']

COLORS_18 = [
    (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
    (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
    (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
    (227, 119, 194), (247, 182, 210), (188, 189, 34), (219, 219, 141),
    (23, 190, 207), (158, 218, 229)
]


def generate_color(color_idx):
    if color_idx < 18:
        return COLORS_18[color_idx]
    else:
        h = (-3.88 * color_idx) % (2 * np.pi)
        if h < 0:
            h += 2 * np.pi
        h /= (2 * np.pi)

        ls = [0.5, 0.6, 0.45, 0.55, 0.35, 0.4]
        l_value = ls[(color_idx // 13) % len(ls)]

        s = 0.4 + 0.2 * np.sin(0.42 * color_idx)
        
        # Convert HSL to RGB
        r, g, b = colorsys.hls_to_rgb(h, l_value, s)
        
        # Return RGB values as a tuple
        return (np.array([r, g, b])*255).astype(int).tolist()


def render_with_view_T(mesh, view_T, img_size=(224, 224), save_dir=False, obj_id=None, **kwargs):
    mesh.apply_transform(view_T)
    
    height, width = img_size
    
    if kwargs.get('data_source', None):
        yfov = get_yfov_from_K(load_K(kwargs['data_source'], scene_name=kwargs['scene_name']), height)
    else:
        yfov = np.pi / 3.0

    cam = PerspectiveCamera(yfov=yfov)
    direc_l = DirectionalLight(color=np.ones(3), intensity=3)
    r = OffscreenRenderer(viewport_width=width, viewport_height=height)
    
    if save_dir:
        assert obj_id is not None
        os.makedirs(save_dir, exist_ok=True)
        mesh.export(os.path.join(save_dir, f'{obj_id}.ply'))
    
    if not isinstance(mesh, trimesh.Scene):
        tri_scene = trimesh.Scene(mesh)
    scene = pyrender.Scene.from_trimesh_scene(tri_scene, bg_color=(0., 0., 0., 0.), ambient_light=[0.2, 0.2, 0.2])
    cam_node = scene.add(cam)
    direc_l_node = scene.add(direc_l)

    # im, _ = r.render(scene, flags=RenderFlags.RGBA | RenderFlags.SHADOWS_ALL)
    im, _ = r.render(scene, flags=RenderFlags.RGBA | RenderFlags.SKIP_CULL_FACES)

    scene.remove_node(cam_node)
    scene.remove_node(direc_l_node)

    return im


def render_pred_scene_from_json(pred_json_path, gt_json_path=None, save_path=None, meshes=None):
    prediction = json.load(open(pred_json_path))
    if gt_json_path is not None:
        gt = json.load(open(gt_json_path))
    
    meshes = meshes if meshes is not None else []
    for obj in prediction["objects"]:
        obj_id = obj["id"]
        if 'retrieval' not in obj:
            continue
        cad_pick = obj['cad_pick'] if 'cad_pick' in obj else 0
        if gt_json_path is not None:
            model_color = gt[str(obj_id)]['model_color']
        else:
            model_color = generate_color(obj_id)
        mesh = load_mesh(obj['retrieval'][cad_pick], normalize=False, base_color=model_color)
        view_trans = np.array(obj["view_trans"])
        if view_trans.ndim == 3:
            view_trans = view_trans[0]
        mesh.apply_transform(view_trans)
        meshes.append(mesh)
    scene_mesh = sum(meshes)
    
    cam = PerspectiveCamera(yfov=np.pi/3)
    direc_l = DirectionalLight(color=np.ones(3), intensity=3)
    spot_l = SpotLight(color=np.ones(3), intensity=15.0, innerConeAngle=np.pi/6, outerConeAngle=np.pi/2)
    r = OffscreenRenderer(viewport_width=1008, viewport_height=784)
    
    if not isinstance(scene_mesh, trimesh.Scene):
        tri_scene = trimesh.Scene(scene_mesh)
    scene = pyrender.Scene.from_trimesh_scene(tri_scene, bg_color=(0., 0., 0., 0.), ambient_light=[0.2, 0.2, 0.2])
    
    cam_node = scene.add(cam)
    direc_l_pose = np.eye(4)
    direc_l_pose[:3, :3] = np.array([[1, 0, 0], [0, np.cos(np.pi/6), np.sin(np.pi/6)], [1, -np.sin(np.pi/6), np.cos(np.pi/6)]])
    direc_l_node = scene.add(direc_l, pose=direc_l_pose)
    spot_l_pose = np.eye(4)
    spot_l_pose[:3, 3] = np.array([0, 0, 1])
    spot_l_node = scene.add(spot_l, pose=spot_l_pose)

    im, _ = r.render(scene, flags=RenderFlags.RGBA | RenderFlags.SHADOWS_ALL)
    # im, _ = r.render(scene, flags=RenderFlags.RGBA | RenderFlags.SHADOWS_ALL | RenderFlags.SKIP_CULL_FACES)

    Image.fromarray(im).save(save_path)

    scene.remove_node(cam_node)
    scene.remove_node(direc_l_node)
    scene.remove_node(spot_l_node)


def render_gt_scene_from_json(gt_json_path, save_path, meshes=None):
    gt = json.load(open(gt_json_path))
    
    meshes = meshes if meshes is not None else []
    for obj_id, obj in gt.items():
        if 'model' not in obj: 
            continue
        mesh = load_mesh(obj['model'], normalize=False, base_color=obj['model_color'])
        mesh.apply_transform(np.array(obj['view_trans']))
        meshes.append(mesh)
    scene_mesh = sum(meshes)
    # scene_mesh.export(save_path.replace('.png', '.obj'))

    cam = PerspectiveCamera(yfov=np.pi/3)
    direc_l = DirectionalLight(color=np.ones(3), intensity=3)
    spot_l = SpotLight(color=np.ones(3), intensity=8.0, innerConeAngle=np.pi/6, outerConeAngle=np.pi/2)
    r = OffscreenRenderer(viewport_width=1008, viewport_height=784)
    
    if not isinstance(scene_mesh, trimesh.Scene):
        tri_scene = trimesh.Scene(scene_mesh)
    scene = pyrender.Scene.from_trimesh_scene(tri_scene, bg_color=(0., 0., 0., 0.), ambient_light=[0.2, 0.2, 0.2])
    
    cam_node = scene.add(cam)
    direc_l_pose = np.eye(4)
    direc_l_pose[:3, :3] = np.array([[1, 0, 0], [0, np.cos(np.pi/6), np.sin(np.pi/6)], [1, -np.sin(np.pi/6), np.cos(np.pi/6)]])
    direc_l_node = scene.add(direc_l, pose=direc_l_pose)
    spot_l_pose = np.eye(4)
    spot_l_pose[:3, 3] = np.array([0, 0, 1])
    spot_l_node = scene.add(spot_l, pose=spot_l_pose)

    im, _ = r.render(scene, flags=RenderFlags.RGBA | RenderFlags.SHADOWS_ALL)

    Image.fromarray(im).save(save_path)

    scene.remove_node(cam_node)
    scene.remove_node(direc_l_node)
    scene.remove_node(spot_l_node)


if __name__ == '__main__':
    from PIL import Image
    cad_entry = "1679,WSS,,8b5a96b72767f7354fac5eaf08c4d9ce,0 0 1,0 -1 0"
    
    wss2scene = np.array([[-0.0254,  0.,      0.,      0.    ],
                        [ 0.,      0.,      0.0254,  0.    ],
                        [ 0.,      0.0254,  0.,      0.    ],
                        [ 0.,      0.,      0.,      1.    ]])
    trs_mat = np.array([1.28221, 0.0, 0.0, 0.0, 0.0, 1.28221, 0.0, 0.0, 0.0, 0.0, 1.28221, 0.0, 57.6319, 223.106, 49.0038, 1.0]).reshape(4, 4).T
    
    cam2world = np.array([-0.7071067811865481,2.7755575615628914e-17,-0.7071067811865469,0,-0.35355339058819135,0.8660254037885882,0.3535533905881921,0,0.612372435698728,0.4999999999928129,-0.6123724356987292,0,-0.07457506359354449,2.209799999999999,3.7775891651840854,1]).reshape(4, 4).T
    world2cam = np.linalg.inv(cam2world)
    
    view_trans = world2cam @ wss2scene @ trs_mat
    
    mesh = load_mesh(cad_entry, normalize=False, load_textures=False)
    im = render_with_view_T(mesh, view_trans, save_dir='./', obj_id='gt_pose')
    Image.fromarray(im).save("gt_pose.png")