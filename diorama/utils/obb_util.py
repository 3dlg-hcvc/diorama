import os, json
from copy import deepcopy
import numpy as np
import trimesh

from diorama.utils.cad_util import parse_cad_entry


SUPP_DIR = {
    "HSSD": "data/fpmodels/support-surfaces",
    "WSS": "data/wss/support-surfaces",
}
CAD_W_SUPP = [cad.strip() for cad in open("data/fpmodels/fpmodels_retrieve_clean_support.csv")]
CAD_W_SUPP.extend([cad.strip() for cad in open("data/wss/wss_models_support.csv")])


def to_hom(mat: np.ndarray) -> np.ndarray:
    is_1d = mat.ndim == 1
    if is_1d:
        mat = mat.reshape(1, -1)
    hom = np.concatenate([mat, np.ones((*mat.shape[:-1], 1))], axis=-1)
    return hom.reshape(-1) if is_1d else hom

def from_hom(mat: np.ndarray) -> np.ndarray:
    is_1d = mat.ndim == 1
    if is_1d:
        mat.reshape(1, -1)
    de_hom, ones = np.split(mat, [3], axis=-1)
    return de_hom.reshape(-1) if is_1d else de_hom


class OBB():
    DEFAULT_CORNERS = np.array([
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ], dtype=float) - 0.5 # Pytorch3d iou calculation expects the corners to be in this order
    DEFAULT_LINES = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [0, 4], [1, 5], [2, 6], [3, 7],
        [4, 5], [5, 6], [6, 7], [7, 4],
    ]
    UP = np.array([0., 1., 0.])
    FRONT = np.array([0., 0., 1.])
    CONTACT_SURF_IDX = [
        [0, 4, 5, 1], [3, 2, 6, 7], # bottom, top
        [0, 3, 7, 4], [1, 5, 6, 2], # left, right
        [0, 1, 2, 3], [4, 7, 6, 5] # back, front
    ]
    CONTACT_SURF_VECS = [
        [0, 1, 0], [0, -1, 0],
        [1, 0, 0], [-1, 0, 0],
        [0, 0, 1], [0, 0, -1]
    ]
    SUPPORT_SURF_IDX = [6, 2, 3, 7]
    
    def __init__(self, corners=None, obb=None, trs_mat=None, norm_mat=None, contact_surfs=None, contact_vecs=None, supp_surfs=None, supp_vecs=None, supp_samples=None, supp_sample_dists=None):
        self.corners = corners
        self.obb = obb
        self.trs_mat = trs_mat
        self.norm_mat = norm_mat
        self.contact_surfs = contact_surfs
        self.contact_vecs = contact_vecs
        self.supp_surfs = supp_surfs
        self.supp_vecs = supp_vecs
        self.supp_samples = supp_samples
        self.supp_sample_dists = supp_sample_dists
        
        self._init_attr()

    def _init_attr(self):
        if self.corners is None:
            self.corners = self.DEFAULT_CORNERS.copy()
        if self.obb is None:
            if self.trs_mat is not None:
                self.decompose_transform()
        if self.trs_mat is None:
            if self.obb is not None:
                self.compose_transform()
            else:
                self.trs_mat = np.identity(4)
                self.decompose_transform()
        if self.contact_surfs is None:
            self.contact_surfs = self.corners[self.CONTACT_SURF_IDX]
            self.contact_vecs = np.array(self.CONTACT_SURF_VECS)
        if self.supp_surfs is None:
            self.supp_surfs = self.corners[self.SUPPORT_SURF_IDX][None, ...]
            self.supp_vecs = self.UP[None, ...]
        if self.supp_samples is None:
            self.supp_samples = self.supp_surfs.mean(1, keepdims=True)
            self.supp_sample_dists = np.array([100.])
        self.sem_front = self.FRONT.copy()
        self.proj_arch = 'floor'

    @classmethod
    def create_from_trimesh(cls, mesh: trimesh.Trimesh, cad_entry: str, norm_mat: np.ndarray = None):
        corners = trimesh.bounds.corners(mesh.bounds) # follow the same corners order
        
        supp_surfs, supp_vecs, supp_samples, supp_sample_dists = [], [], [], []
        if cad_entry in CAD_W_SUPP:
            _, dataset, subdir, cad_id, _, _ = parse_cad_entry(cad_entry)
            support_json_path = f"{SUPP_DIR[dataset]}/{cad_id}/{cad_id}.supportSurface.json"
            if not os.path.exists(support_json_path):
                return cls(corners=corners, norm_mat=norm_mat)
            support_json = json.load(open(support_json_path))
            model2world = np.array(support_json["modelToWorld"]).reshape(4,4).T
            world2model = np.linalg.inv(model2world)
            support_surf_candis = support_json["supportSurfaces"]
            for supp_surf in support_surf_candis:
                # if not supp_surf["isHorizontal"] or not supp_surf["isVertical"]: 
                if not (supp_surf["isHorizontal"] or supp_surf["isVertical"]): 
                    continue
                if supp_surf["isVertical"] and supp_surf["isInterior"]:
                    continue
                surf_normal = np.array(list(supp_surf["normal"].values()))
                surf_normal = from_hom(to_hom(surf_normal) @ world2model.T)
                if norm_mat is not None:
                    surf_normal = norm_mat[:3, :3] @ surf_normal
                    surf_normal = surf_normal / np.linalg.norm(surf_normal)
                if 1-surf_normal[1] > 1e-5 and supp_surf["isHorizontal"]: 
                    continue # +y up
                
                default_corners = cls.DEFAULT_CORNERS.copy()
                surf_obb = supp_surf["obb"]
                surf_obb_corners = (default_corners * np.array(surf_obb["axesLengths"])) @ np.array(surf_obb["normalizedAxes"]).reshape(3,3).T + np.array(surf_obb["centroid"])
                surf_obb_corners = from_hom(to_hom(surf_obb_corners) @ world2model.T)
                if norm_mat is not None:
                    surf_obb_corners = surf_obb_corners @ norm_mat[:3, :3].T + norm_mat[:3, 3].T
                
                # surf_obb_corners = (surf_obb_corners[[0,1,5,4]] + surf_obb_corners[[3,2,6,7]]) / 2
                min_len_idx = np.argmin(surf_obb["axesLengths"])
                if min_len_idx == 1:
                    surf_obb_corners = (surf_obb_corners[[0,1,5,4]] + surf_obb_corners[[3,2,6,7]]) / 2
                elif min_len_idx == 2:
                    surf_obb_corners = (surf_obb_corners[[0,1,2,3]] + surf_obb_corners[[4,5,6,7]]) / 2
                else:
                    surf_obb_corners = (surf_obb_corners[[0,3,7,4]] + surf_obb_corners[[1,2,6,5]]) / 2
                    
                supp_two_bound = surf_obb_corners[1:3] - surf_obb_corners[:2] # (2,3)
                surf_bound_lens = np.linalg.norm(supp_two_bound, axis=-1) # (2,)
                if np.any(surf_bound_lens < 0.05): 
                    continue
                supp_surfs.append(surf_obb_corners)
                supp_vecs.append(surf_normal)
                
                supp_surf_samples, supp_surf_sample_dists = [], []
                for sp in supp_surf["samples"]:
                    point = np.array(sp["point"])
                    point = from_hom(to_hom(point) @ world2model.T)
                    if norm_mat is not None:
                        point = point @ norm_mat[:3, :3].T + norm_mat[:3, 3].T
                    supp_surf_samples.append(point)
                    supp_surf_sample_dists.append(sp["clearance"] if "clearance" in sp else 100)
                supp_samples.append(supp_surf_samples)
                supp_sample_dists.append(supp_surf_sample_dists)
            supp_surfs = np.array(supp_surfs)
            supp_vecs = np.array(supp_vecs)
            supp_samples = np.array(supp_samples)
            supp_sample_dists = np.array(supp_sample_dists)
        
        if len(supp_surfs) == 0:
            supp_surfs = None
            supp_vecs = None
            supp_samples = None
            supp_sample_dists = None
        
        return cls(corners=corners, norm_mat=norm_mat, supp_surfs=supp_surfs, supp_vecs=supp_vecs, supp_samples=supp_samples, supp_sample_dists=supp_sample_dists)
    
    @classmethod
    def create_from_json(cls, json_path):
        obb_json = json.load(open(json_path))
        
        corners = np.array(obb_json["corners"])
        norm_mat = np.array(obb_json["norm_mat"])
        trs_mat = np.array(obb_json["trs_mat"])
        contact_surfs = np.array(obb_json["contact_surfs"])
        contact_vecs = np.array(obb_json["contact_vecs"])
        supp_surfs = np.array(obb_json["supp_surfs"])
        supp_vecs = np.array(obb_json["supp_vecs"])
        supp_samples = np.array(obb_json["supp_samples"]) if "supp_samples" in obb_json else None
        supp_sample_dists = np.array(obb_json["supp_sample_dists"]) if "supp_sample_dists" in obb_json else None
        
        return cls(corners=corners, trs_mat=trs_mat, norm_mat=norm_mat, contact_surfs=contact_surfs, contact_vecs=contact_vecs, supp_surfs=supp_surfs, supp_vecs=supp_vecs, supp_samples=supp_samples, supp_sample_dists=supp_sample_dists)
    
    def get_corners(self):
        corners = from_hom(to_hom(self.corners) @ self.trs_mat.T)
        return corners
    
    def get_surfs(self):
        num_supp_surfs = len(self.supp_surfs)
        supp_surfs_corners = from_hom(to_hom(self.supp_surfs.reshape(-1, 3)) @ self.trs_mat.T).reshape(num_supp_surfs, 4, 3)
        num_contact_surfs = len(self.contact_surfs)
        contact_surfs_corners = from_hom(to_hom(self.contact_surfs.reshape(-1, 3)) @ self.trs_mat.T).reshape(num_contact_surfs, 4, 3)
        surfs = {
            "contact": contact_surfs_corners,
            "support": supp_surfs_corners,
        }
        return surfs
    
    def get_vecs(self):
        contact_vecs = self.contact_vecs @ self.obb["rotation"].T
        supp_vecs = self.supp_vecs @ self.obb["rotation"].T
        sem_front = self.sem_front @ self.obb["rotation"].T
        vecs = {
            "contact": contact_vecs,
            "support": supp_vecs,
            "sem_front": sem_front,
        }
        return vecs
    
    def get_supp_samples(self):
        # num_supp_surfs = len(self.supp_samples)
        supp_samples = from_hom(to_hom(self.supp_samples.reshape(-1, 3)) @ self.trs_mat.T)#.reshape(num_supp_surfs, -1, 3)
        return supp_samples
        
    
    def apply_transform(self, trs_mat):
        self.trs_mat = trs_mat
        self.decompose_transform()
        
    def decompose_transform(self):
        scale = np.linalg.norm(self.trs_mat[:3, :3], axis=0)
        rotation = self.trs_mat[:3, :3] / scale[None, ...]
        translation = self.trs_mat[:3, 3]
        self.obb = {
            "scale": scale,
            "rotation": rotation,
            "translation": translation,
        }
        
    def compose_transform(self):
        trs_mat = np.eye(4)
        trs_mat[np.arange(3), np.arange(3)] = self.obb["scale"]
        trs_mat[:3, :3] = self.obb["rotation"] @ trs_mat[:3, :3]
        trs_mat[:3, 3] = self.obb["translation"]
        self.trs_mat = trs_mat
        
    def redefine_surfs_from_relation(self, supp_type):
        if supp_type in ["mounted on"]:
            self.proj_arch = 'wall'
            self.sem_front = self.UP.copy()
            # if len(self.supp_surfs) <= 1:
            #     supp_surf_idx = [4, 5, 6, 7]
            #     self.supp_surfs = self.corners[supp_surf_idx][None, ...]
            #     self.supp_vecs = self.FRONT[None, ...]
            return
            contact_surf_idx = [0, 1, 2, 3]
            self.contact_surfs = self.corners[contact_surf_idx][None, ...]
            self.contact_vecs = self.FRONT[None, ...]
    
    def find_closest_supp_surf(self, other_obb, supp_type=None, visual_path=False):
        # other_contact_surfs = other_obb.get_surfs()["contact"] # (n, 4, 3)
        # other_contact_ctrs = other_contact_surfs.mean(1) # (n, 3)
        other_obb_corners = other_obb.get_corners() # (8, 3)
        if supp_type in ["mounted on"] or supp_type is None:
            anchor_point = other_obb_corners.mean(0)
        else: # take gravity direction
            anchor_point = other_obb_corners[np.argmin(other_obb_corners[:,1])]
        
        if supp_type is None:
            valid_supp_indices = np.arange(len(self.supp_vecs))
        elif supp_type in ["mounted on"]:
            valid_supp_indices = np.where(np.abs(self.supp_vecs[:, 1]) < 0.2)[0]
        else:
            valid_supp_indices = np.where(np.abs(self.supp_vecs[:, 1]) > 0.8)[0]
        if len(valid_supp_indices) == 0:
            valid_supp_indices = np.arange(len(self.supp_vecs))
        
        support_surfs = self.get_surfs()["support"][valid_supp_indices] # (k, 4, 3)
        support_samples = self.get_supp_samples().reshape(len(self.supp_vecs), -1, 3)[valid_supp_indices].reshape(-1, 3) # (k*m, 3)
        
        # For each surface, compute minimal distance from anchor point to the sample points
        all_supp_sample_dists = np.linalg.norm(anchor_point[None, ...] - support_samples, axis=-1).reshape(len(support_surfs), -1)
        min_supp_sample_dists = np.min(all_supp_sample_dists, axis=-1)
        supp_idx = np.argmin(min_supp_sample_dists)
        supp_idx = valid_supp_indices[supp_idx]
        
        return supp_idx, self.supp_surfs[supp_idx], self.supp_vecs[supp_idx], self.supp_samples[supp_idx], self.supp_sample_dists[supp_idx]
    
    def find_closest_contact_surf(self, other_obb, other_surf_idx=0, visual_path=False):
        other_support = other_obb.get_surfs()["support"][other_surf_idx] # (4, 3)
        other_support_vec = other_obb.get_vecs()["support"][other_surf_idx] # (3,)
        contact_surfs = self.get_surfs()["contact"] # (n, 4, 3)
        contact_vecs = self.get_vecs()["contact"] # (n, 3)
        
        contact_vecs[0] *= 1.44 # HACK: put more weights on the bottom surface, basically choose the bottom surface if angle is less than 55 degree
        
        min_vd_ind = np.argsort(np.dot(contact_vecs, other_support_vec))[::-1][:3] # take top 3
        contact_idx = min_vd_ind[0]
        # supp_surf_dists = np.abs(((contact_surf_ctrs[min_vd_ind] - other_support[0]) * other_support_vec).sum(1)) # (n,)
        # min_sd_idx = np.argmin(supp_surf_dists)
        # contact_idx = min_vd_ind[min_sd_idx]
        
        if visual_path:
            obb_m1 = self.export_obb_mesh()
            contact_surf_mesh = trimesh.Trimesh(vertices=contact_surfs[contact_idx], faces=[[0,1,2], [0,2,3]], vertex_colors=[0,0,1,0.1], face_colors=[0,0,1,0.1])
            obb_m2 = other_obb.export_obb_mesh()
            supp_surf_mesh = trimesh.Trimesh(vertices=other_support, faces=[[0,1,2], [0,2,3]] if len(other_support) == 4 else [[0,1,2]], vertex_colors=[0,1,0,0.1], face_colors=[0,1,0,0.1])
            mesh = sum([obb_m1, contact_surf_mesh, obb_m2, supp_surf_mesh])
            mesh.export(visual_path)
        
        return contact_idx, self.contact_surfs[contact_idx], self.contact_vecs[contact_idx]
    
    def find_closest_adhere_surf(self, other_obb, other_surf_idx=0, adhere_thres=0.5, visual_path=False):
        other_support = other_obb.get_surfs()["support"][other_surf_idx] # (4, 3)
        other_support_vec = other_obb.get_vecs()["support"][other_surf_idx] # (3,)
        contact_surfs = self.get_surfs()["contact"] # (n, 4, 3)
        contact_surf_ctrs = contact_surfs.mean(1) # (n, 3)
        contact_vecs = self.get_vecs()["contact"] # (n, 3)
        
        min_vd_ind = np.argsort(np.dot(contact_vecs, other_support_vec))[::-1] # take top 3
        supp_surf_dist = np.abs(((contact_surf_ctrs[min_vd_ind] - other_support[0]) * other_support_vec).sum(1))[0] # (n,)
        adhered = supp_surf_dist <= adhere_thres
        adhere_idx = min_vd_ind[0]
        # contact_idx = min_vd_ind[min_sd_idx]
        
        if adhered:
            if visual_path:
                obb_m1 = self.export_obb_mesh()
                contact_surf_mesh = trimesh.Trimesh(vertices=contact_surfs[adhere_idx], faces=[[0,1,2], [0,2,3]], vertex_colors=[0,0,1,0.1], face_colors=[0,0,1,0.1])
                obb_m2 = other_obb.export_obb_mesh()
                supp_surf_mesh = trimesh.Trimesh(vertices=other_support, faces=[[0,1,2], [0,2,3]] if len(other_support) == 4 else [[0,1,2]], vertex_colors=[0,1,0,0.1], face_colors=[0,1,0,0.1])
                mesh = sum([obb_m1, contact_surf_mesh, obb_m2, supp_surf_mesh])
                mesh.export(visual_path)
            
            return adhered, self.contact_surfs[adhere_idx], self.contact_vecs[adhere_idx]
        else:
            return adhered, None, None
    
    
    def export_optimization(self):
        optim_obb = {
            "corners": self.corners.copy(),
            "center": self.corners.mean(0),
            # "contact_surf": self.contact_surfs[0],
            # "contact_vec": self.contact_vecs[0],
            "sem_front": self.sem_front.copy()
        }
        optim_obb.update(deepcopy(self.obb))
        return optim_obb
    
    def export_json(self, filename=None, return_dict=False):
        obb_json = {
            "corners": self.corners.tolist(), # before transform
            "norm_mat": self.norm_mat.tolist(),
            "trs_mat": self.trs_mat.tolist(),
            "contact_surfs": self.contact_surfs.tolist(),
            "contact_vecs": self.contact_vecs.tolist(),
            "supp_surfs": self.supp_surfs.tolist(),
            "supp_vecs": self.supp_vecs.tolist(),
            "supp_samples": self.supp_samples.tolist(),
            "supp_sample_dists": self.supp_sample_dists.tolist(),
        }
        if filename is not None:
            with open(filename, "w") as f:
                json.dump(obb_json, f, indent=4)
        if return_dict:
            return obb_json
    
    def export_obb_mesh(self, color=None, radius=0.01, texture=False):
        corners = self.get_corners()
        obb_mesh = []
        for idx1, idx2 in self.DEFAULT_LINES:
            line = corners[idx1], corners[idx2]
            line_mesh = trimesh.creation.cylinder(radius, sections=8, segment=line)
            obb_mesh.append(line_mesh)
        obb_mesh = sum(obb_mesh)
        if color is not None:
            obb_mesh = self.colorize_mesh(obb_mesh, color, texture)
        return obb_mesh
    
    def export_obb_mesh_w_surfs(self, obb_color=None, obb_radius=0.01):
        obb_mesh = self.export_obb_mesh(obb_color, obb_radius)
        surfs = self.get_surfs()
        for surf_type, surf_corners in surfs.items():
            if surf_type == 'contact': continue
            plane_color = [1,0,0,0.1] if surf_type == 'contact' else [0,1,0,0.1]
            for i in range(len(surf_corners)):
                plane_mesh = trimesh.Trimesh(vertices=surf_corners[i], faces=[[0,1,2], [0,2,3]], vertex_colors=plane_color, face_colors=plane_color)
                obb_mesh += plane_mesh
        return obb_mesh
    
    @staticmethod
    def colorize_mesh(mesh, color, texture=False):
        mesh.visual.vertex_colors[:] = np.append(color, 255).astype(np.uint8)
        if texture:
            # mesh.vertex_normals
            mesh.visual = mesh.visual.to_texture()
            mesh.visual.uv = np.random.rand(*mesh.visual.uv.shape)
        return mesh
    
    
if __name__ == '__main__':
    from diorama.utils.cad_util import CAD_DIR, load_mesh
    # cad_entry = "14348,HSSD,d,d213d8ab611456a8c8d1f8b1f76bd075d9070f48,,"
    # cad_entry = "7697,HSSD,6,6386fffd6df3a42edd6e0a3567425d58ef5455a4,,"
    # cad_entry = "13298,HSSD,c,c075ced257f48c753d22d3bd3400186d6de319da,,"
    # cad_entry = "9287,HSSD,7,7d1f2631bececebeb2ec101b35ab998fe9cc1506,,"
    # cad_entry = "1123,WSS,,67fb05261fbd0048643bc5e60b471691,0 0 1,0 -1 0"
    # cad_entry = "125,WSS,,4be0f809a66f1797ad9d8182e90fd7fb,0 0 1,0 -1 0"
    cad_entry = "538,WSS,,ee9136090b140ace3f7a74e12a274ef,0 0 1,0 -1 0"
    # cad_entry = "1634,WSS,,c12495749f34cc9013d3eba49e0a8940,0 0 1,0 -1 0"
    # cad_entry = "598,WSS,,a728186f2bb912572d8564c06b061019,0 0 1,0 -1 0"

    mesh, norm_mat = load_mesh(cad_entry, normalize=True)
    # mesh = load_mesh(cad_entry, normalize=False)
    
    obb = OBB.create_from_trimesh(mesh, cad_entry, norm_mat=norm_mat)
    # obb = OBB.create_from_trimesh(mesh, cad_entry)
    
    obb_mesh = obb.export_obb_mesh_w_surfs()
    obb_mesh.export('obb5.ply')
    obb.export_json('obb5.json')
    
    # # obb = OBB.create_from_json(json.load(open(f"output/0034/cad/6.obb.json")))
    # obb = OBB.create_from_json(json.load(open(f"output/0104/cad/21.obb.json")))
    # obb_mesh = obb.export_obb_mesh_w_surfs()
    # obb_mesh.export('obb4.ply')
    
    # # _, dataset, subdir, cad_id, _, _ = "10975,HSSD,9,99714c9a305551cf4493ab7087c60fd6e7cba4ea,,".split(',')
    # obb1 = OBB.create_from_json(json.load(open(f"output/0247/cad/1.obb.json")))
    # obb2 = OBB.create_from_json(json.load(open(f"output/0247/cad/4.obb.json")))
    # supp_idx, _, _, _ = obb2.find_closest_supp_surf(obb1)
    