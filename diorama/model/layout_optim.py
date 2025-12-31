import os
import json
import numpy as np
import torch
from pytorch3d.transforms import matrix_to_euler_angles, euler_angles_to_matrix

from diorama.scenespec import SceneSpec
from diorama.utils.cad_util import load_mesh
from diorama.utils.obb_util import OBB
from diorama.utils.arch_util import create_arch_plane, load_architecture_from_path
from diorama.utils.basic_util import list_of_dict_to_dict_of_array, dict_of_array_to_list_of_dict, recursively_to
# from diorama.utils.render_util import render_with_view_T


class LayoutOptimizer:
    
    OBB_KEYS = ["scale", "rotation", "translation"]
    
    def __init__(self, scenespec: SceneSpec, output_dir, query_dir, arch_dir, arch_mask_dir=None, cad_picks=None) -> None:
        self.scenespec = scenespec
        self.output_dir = output_dir
        self.query_dir = query_dir
        self.arch_dir = arch_dir
        self.arch_mask_dir = arch_mask_dir
        self.arch_elements = scenespec.arch_elements
        
        objects = {}
        self.obbs = []
        if cad_picks is not None:
            self.cad_picks = cad_picks
        else:
            self.cad_picks = [0] * len(scenespec.objects)
        for obj_id, cad_pick in zip(scenespec.objects.keys(), self.cad_picks):
            if int(cad_pick) == -1:
                continue
            obb_path = f"{query_dir}/cad/{obj_id}.{cad_pick}.obb.json"
            if not os.path.exists(obb_path):
                obb_path = f"{query_dir}/cad/{obj_id}.obb.json"
            self.obbs.append(OBB.create_from_json(obb_path))
            objects[obj_id]= scenespec.objects[obj_id]
            objects[obj_id]["cad_pick"] = cad_pick
        
        self.id2idx = {obj_id: idx for idx, obj_id in enumerate(list(objects.keys()))}
        self.idx2id = {v:k for k,v in self.id2idx.items()}
        self.objects = objects
        self.RELATION = {
            "support": [rel for rel in scenespec.support_relations if rel[1] in self.objects or (rel[1] in self.arch_elements and rel[0] in self.objects)],
        }
        
        if arch_dir is not None:
            self.load_prebuilt_archs()
        else:
            self.build_archs()
        self.identify_support_surfaces()
        self.identify_adherence_surfaces()
        self.to_optimization()
    
    
    def build_archs(self):
        self.archs = {}
        for arch_id, arch in self.arch_elements.items():
            arch_pcd_file = f"{self.query_dir}/pcd/pcd.{arch_id}.ply"
            arch_plane = create_arch_plane(arch["name"], arch_pcd_file)
            arch_plane.instantiate_arch_element()
            arch_plane.extrude_arch_element()
            self.archs[arch_id] = arch_plane
    

    def load_prebuilt_archs(self):
        # if "wss" in self.arch_dir:
        #     self.archs = load_wss_architecture_from_path(self.arch_dir.replace("SCENE", os.path.basename(self.query_dir)))
        # else:
        self.archs = load_architecture_from_path(self.arch_dir.replace("SCENE", os.path.basename(self.query_dir)), self.scenespec.arch_elements, self.arch_mask_dir)
            # self.archs[24].arch_normal = -self.archs[24].arch_normal
    
    
    def identify_support_surfaces(self):
        self.RELATION["support"] = sorted(self.RELATION["support"], key=lambda r: r[0])
        
        for supp_rel in self.RELATION["support"]:
            subj_id, tgt_id, rel_type = supp_rel
            self.obbs[self.id2idx[subj_id]].redefine_surfs_from_relation(rel_type)
            # if tgt_id in self.archs and self.archs[tgt_id].arch_type == "wall":
            #     contact_proj_wall = (self.obbs[self.id2idx[subj_id]].get_vecs()["contact"][0] * self.archs[tgt_id].arch_normal).sum()
            #     if contact_proj_wall < 0:
            #         inverse_sem_mat = np.array([[-1,0,0], [0,1,0], [0,0,-1]])
            #         obb_trs_mat = self.obbs[self.id2idx[subj_id]].trs_mat.copy()
            #         obb_trs_mat[:3, :3] = obb_trs_mat[:3, :3] @ inverse_sem_mat
            #         self.obbs[self.id2idx[subj_id]].apply_transform(obb_trs_mat)
        
        self.paired_supp_idx = [-1] * len(self.obbs)
        self.paired_supp_surfs = np.zeros((len(self.obbs), 4, 3))
        self.paired_supp_vecs = np.zeros((len(self.obbs), 3))
        self.paired_supp_samples = np.zeros((len(self.obbs), 100, 3))
        self.paired_supp_sample_dists = np.zeros((len(self.obbs), 100))
        self.paired_contact_surfs = np.zeros((len(self.obbs), 4, 3))
        self.paired_contact_vecs = np.zeros((len(self.obbs), 3))
        for supp_rel in self.RELATION["support"]:
            subj_id, tgt_id, rel_type = supp_rel
            if tgt_id in self.arch_elements: 
                if tgt_id not in self.archs:
                    continue
                _, contact_surf, contact_vec = self.obbs[self.id2idx[subj_id]].find_closest_contact_surf(self.archs[tgt_id], 0, visual_path=f"{self.output_dir}/cad/{subj_id}.supported.ply")
            else:
                self.paired_supp_idx[self.id2idx[subj_id]] = self.id2idx[tgt_id]
                supp_idx, supp_surf, supp_vec, supp_samples, supp_sample_dists = self.obbs[self.id2idx[tgt_id]].find_closest_supp_surf(self.obbs[self.id2idx[subj_id]], rel_type)
                self.scenespec.add_support_surf(subj_id, tgt_id, supp_idx, supp_surf, supp_vec)
                self.paired_supp_surfs[self.id2idx[subj_id]] = supp_surf
                self.paired_supp_vecs[self.id2idx[subj_id]] = supp_vec
                self.paired_supp_samples[self.id2idx[subj_id]] = supp_samples
                self.paired_supp_sample_dists[self.id2idx[subj_id]] = supp_sample_dists
                _, contact_surf, contact_vec = self.obbs[self.id2idx[subj_id]].find_closest_contact_surf(self.obbs[self.id2idx[tgt_id]], supp_idx, visual_path=f"{self.output_dir}/cad/{subj_id}.supported.ply")
            self.paired_contact_surfs[self.id2idx[subj_id]] = contact_surf
            self.paired_contact_vecs[self.id2idx[subj_id]] = contact_vec
    
    
    def identify_adherence_surfaces(self):
        self.paired_adhere_idx = []
        self.paired_adhere_arch_ids = []
        paired_adhere_surfs = []
        paired_adhere_vecs = []
        for supp_rel in self.RELATION["support"]:
            subj_id, tgt_id, _ = supp_rel
            if tgt_id not in self.arch_elements:
                continue
            for arch_id in self.archs:
                if arch_id == tgt_id:
                    continue
                adhered, adhere_surf, adhere_vec = self.obbs[self.id2idx[subj_id]].find_closest_adhere_surf(self.archs[arch_id], visual_path=f"{self.output_dir}/cad/{subj_id}.adhere.ply")
                if adhered:
                    self.paired_adhere_idx.append(self.id2idx[subj_id])
                    self.paired_adhere_arch_ids.append(arch_id)
                    paired_adhere_surfs.append(adhere_surf)
                    paired_adhere_vecs.append(adhere_vec)
        self.paired_adhere_surfs = np.array(paired_adhere_surfs)
        self.paired_adhere_vecs = np.array(paired_adhere_vecs)
    
    
    def to_optimization(self):
        obbs_export = [obb.export_optimization() for obb in self.obbs]
        pre_obbs = list_of_dict_to_dict_of_array(obbs_export, to_tensor=True)
        pre_obbs['supp_surf'] = torch.tensor(self.paired_supp_surfs, dtype=torch.float)
        pre_obbs['supp_vec'] = torch.tensor(self.paired_supp_vecs, dtype=torch.float)
        pre_obbs['supp_samples'] = torch.tensor(self.paired_supp_samples, dtype=torch.float)
        pre_obbs['supp_dists'] = torch.tensor(self.paired_supp_sample_dists, dtype=torch.float)
        pre_obbs['contact_surf'] = torch.tensor(self.paired_contact_surfs, dtype=torch.float)
        pre_obbs['contact_vec'] = torch.tensor(self.paired_contact_vecs, dtype=torch.float)
        
        pre_obbs['adhere_surf'] = torch.tensor(self.paired_adhere_surfs, dtype=torch.float)
        pre_obbs['adhere_vec'] = torch.tensor(self.paired_adhere_vecs, dtype=torch.float)

        archs_export = {arch_id: arch.export_optimization() for arch_id, arch in self.archs.items()}
        self.archs_export = recursively_to(archs_export, dtype="tensor")
        
        return pre_obbs
    
    
    def support_align_loss(self, optim_rot, optim_obbs, supp_rels):
        supp_vec, contact_vec = optim_obbs["supp_vec"], optim_obbs["contact_vec"]
        optim_supp_vec = (optim_rot[self.paired_supp_idx].detach() @ supp_vec[..., None]).squeeze(2)
        optim_contact_vec = (optim_rot @ contact_vec[..., None]).squeeze(2)
        
        align_loss = torch.tensor(0.0, requires_grad=True)
        for subj_id, tgt_id, _ in supp_rels:
            if tgt_id in self.arch_elements:
                if tgt_id not in self.archs:
                    continue
                supp_dir_loss = torch.norm(optim_contact_vec[self.id2idx[subj_id]] - self.archs_export[tgt_id]["supp_vec"])
            else:
                supp_dir_loss = torch.norm(optim_contact_vec[self.id2idx[subj_id]] - optim_supp_vec[self.id2idx[subj_id]])
            align_loss = align_loss + supp_dir_loss
        
        if len(self.paired_adhere_idx) > 0:
            optim_adhere_vec = optim_obbs['adhere_vec'] # (m, 3)
            optim_adhere_vec = (optim_rot[self.paired_adhere_idx] @ optim_adhere_vec[..., None]).squeeze(2)
            for idx, arch_id in enumerate(self.paired_adhere_arch_ids):
                adhere_dir_loss = torch.norm(optim_adhere_vec[idx] - self.archs_export[arch_id]["supp_vec"])
                align_loss = align_loss + 5*adhere_dir_loss
            
        return align_loss
    
    
    def place_on_loss(self, optim_obbs, supp_rels):
        optim_scale = optim_obbs['scale'].detach()
        optim_rot = optim_obbs['rotation'].detach()
        optim_translation = optim_obbs['translation']

        optim_contact_ctr = optim_obbs['contact_surf'].mean(dim=1) # (batch, 3)
        optim_contact_ctr = torch.einsum("bij,bkj->bi", optim_rot, (optim_scale * optim_contact_ctr).unsqueeze(1)) + optim_translation # (batch, 3)
        optim_supp_surf = optim_obbs['supp_surf'] # (n, 4, 3)
        optim_supp_surf = torch.einsum("bij,bnj->bni", optim_rot[self.paired_supp_idx], (optim_scale[self.paired_supp_idx].unsqueeze(1) * optim_supp_surf)) + optim_translation[self.paired_supp_idx].unsqueeze(1) # (n, 4, 3)
        optim_supp_vec = optim_obbs['supp_vec'] # (n, 3)
        optim_supp_vec = (optim_rot[self.paired_supp_idx] @ optim_supp_vec[..., None]).squeeze(2)

        place_loss = torch.tensor(0.0, requires_grad=True)
        for subj_id, tgt_id, _ in supp_rels:
            if tgt_id in self.arch_elements:
                if tgt_id not in self.archs:
                    continue
                sub_dist_loss = torch.abs(((optim_contact_ctr[self.id2idx[subj_id]] - self.archs_export[tgt_id]["corners"][0]) * self.archs_export[tgt_id]["supp_vec"]).sum())
            else:
                sub_dist_loss = torch.abs(((optim_contact_ctr[self.id2idx[subj_id]] - optim_supp_surf[self.id2idx[subj_id],0,:]) * optim_supp_vec[self.id2idx[subj_id]]).sum())
            place_loss = place_loss + sub_dist_loss
        
        if len(self.paired_adhere_idx) > 0:
            optim_adhere_ctr = optim_obbs['adhere_surf'].mean(dim=1) # (m, 3)
            optim_adhere_ctr = torch.einsum("bij,bkj->bi", optim_rot[self.paired_adhere_idx], (optim_scale[self.paired_adhere_idx] * optim_adhere_ctr).unsqueeze(1)) + optim_translation[self.paired_adhere_idx] # (batch, 3)
            for idx, arch_id in enumerate(self.paired_adhere_arch_ids):
                adhere_dist_loss = torch.abs(((optim_adhere_ctr[idx] - self.archs_export[arch_id]["corners"][0]) * self.archs_export[arch_id]["supp_vec"]).sum())
                place_loss = place_loss + adhere_dist_loss
            
        return place_loss / len(supp_rels)
    
    
    def adherence_loss(self, optim_obbs):
        optim_scale = optim_obbs['scale'].detach()
        optim_rot = optim_obbs['rotation']
        optim_translation = optim_obbs['translation']
        
        optim_adhere_vec = optim_obbs['adhere_vec'] # (m, 3)
        optim_adhere_vec = (optim_rot[self.paired_adhere_idx] @ optim_adhere_vec[..., None]).squeeze(2)
        optim_adhere_ctr = optim_obbs['adhere_surf'].mean(dim=1) # (m, 3)
        optim_adhere_ctr = torch.einsum("bij,bkj->bi", optim_rot[self.paired_adhere_idx], (optim_scale[self.paired_adhere_idx] * optim_adhere_ctr).unsqueeze(1)) + optim_translation[self.paired_adhere_idx] # (batch, 3)
        
        adhere_loss = torch.tensor(0.0, requires_grad=True)
        for idx, arch_id in enumerate(self.paired_adhere_arch_ids):
            adhere_dir_loss = torch.norm(optim_adhere_vec[idx] - self.archs_export[arch_id]["supp_vec"])
            adhere_dist_loss = torch.abs(((optim_adhere_ctr[idx] - self.archs_export[arch_id]["corners"][0]) * self.archs_export[arch_id]["supp_vec"]).sum())
            adhere_loss = adhere_loss + (adhere_dir_loss + adhere_dist_loss)
        
        return adhere_loss / len(self.paired_adhere_arch_ids)


    def support_space_loss(self, optim_obbs, supp_rels):
        XZ_PLANE = torch.tensor([1, 0, 1]).float()
        XY_PLANE = torch.tensor([1, 1, 0]).float()
        
        optim_scale = optim_obbs['scale']
        optim_rot = optim_obbs['rotation'].detach() # (batch, 3, 3)
        optim_translation = optim_obbs['translation']

        optim_contact_surf = optim_obbs['contact_surf'] # (batch, 4, 3)
        optim_contact_surf = torch.einsum("bij,bnj->bni", optim_rot, (optim_scale.detach().unsqueeze(1) * optim_contact_surf)) + optim_translation.detach().unsqueeze(1) # (batch, 4, 3)
        optim_supp_surf = optim_obbs['supp_surf'] # (batch, 4, 3)
        # optim_supp_surf = torch.einsum("bij,bnj->bni", optim_rot, (optim_scale.unsqueeze(1) * optim_supp_surf)) + optim_translation.unsqueeze(1) # (batch, 4, 3)
        optim_supp_surf = torch.einsum("bij,bnj->bni", optim_rot[self.paired_supp_idx], (optim_scale[self.paired_supp_idx].unsqueeze(1) * optim_supp_surf)) + optim_translation[self.paired_supp_idx].unsqueeze(1) # (n, 4, 3)
        
        supp2all = {}
        for subj_id, tgt_id, _ in supp_rels:
            supp2all.setdefault(tgt_id, [])
            supp2all[tgt_id].append(self.id2idx[subj_id])
        
        space_loss = torch.tensor(0.0, requires_grad=True)
        for tgt_id, all_support in supp2all.items():
            if tgt_id in self.arch_elements:
                continue
            if self.obbs[self.id2idx[tgt_id]].proj_arch == 'floor':
                proj_plane = XZ_PLANE
            else:
                proj_plane = XY_PLANE
            
            optim_contact_surf_proj = optim_contact_surf[all_support] * proj_plane.view(1, 1, 3) # (m, 4, 3)
            optim_contact_surf_proj = optim_contact_surf_proj.reshape(-1, 3).expand(2, -1, -1) # (2, m*4, 3)
            optim_supp_surf_proj = optim_supp_surf[all_support[0]] * proj_plane # (4,3)
            optim_supp_surf_proj_ctr = optim_supp_surf_proj.mean(0).view(1, 1, 3) # (1, 1, 3)
            supp_two_bound = optim_supp_surf_proj[1:3] - optim_supp_surf_proj[:2] # (2,3)
            supp_bound_lens = torch.norm(supp_two_bound, dim=-1, keepdim=True) # (2,1)
            axes = (supp_two_bound / supp_bound_lens).view(2, 1, 3) # (2,1,3)
            # axes = optim_obbs['rotation'].permute(2, 0, 1)[[], id2idx[subj_id], :] # (3,n,3)
            contact_proj_axes = torch.abs((axes * (optim_contact_surf_proj - optim_supp_surf_proj_ctr)).sum(axis=-1)) # (2,m*4)
            subj_space_loss = torch.relu(contact_proj_axes - supp_bound_lens / 2).mean() # (2,m*4)
            
            space_loss = space_loss + subj_space_loss
            
        return space_loss


    def support_volume_loss(self, optim_obbs, supp_rels):
        optim_scale = optim_obbs['scale']
        optim_rot = optim_obbs['rotation'].detach() # (batch, 3, 3)
        optim_translation = optim_obbs['translation']
        
        optim_corners = torch.einsum("bij,bnj->bni", optim_rot, (optim_scale.unsqueeze(1) * optim_obbs['corners'])) + optim_translation.unsqueeze(1) # (n,8,3) corners
        optim_contact_surf = optim_obbs['contact_surf'] # (batch, 4, 3)
        optim_contact_surf = torch.einsum("bij,bnj->bni", optim_rot, (optim_scale.unsqueeze(1) * optim_contact_surf)) + optim_translation.unsqueeze(1) # (batch, 4, 3)
        
        optim_supp_surf = optim_obbs['supp_surf'] # (batch, 4, 3)
        optim_supp_surf = torch.einsum("bij,bnj->bni", optim_rot[self.paired_supp_idx], (optim_scale[self.paired_supp_idx].unsqueeze(1) * optim_supp_surf)) + optim_translation[self.paired_supp_idx].unsqueeze(1) # (n, 4, 3)
        optim_supp_surf = optim_supp_surf.detach()
        
        optim_supp_vec = optim_obbs['supp_vec'] # (n, 3)
        optim_supp_vec = (optim_rot[self.paired_supp_idx] @ optim_supp_vec[..., None]).squeeze(2)
        
        optim_supp_samples = optim_obbs['supp_samples'] # (n, 100, 3)
        optim_supp_samples = torch.einsum("bij,bnj->bni", optim_rot[self.paired_supp_idx], (optim_scale[self.paired_supp_idx].unsqueeze(1) * optim_supp_samples)) + optim_translation[self.paired_supp_idx].unsqueeze(1) # (n, 100, 3)
        optim_supp_samples = optim_supp_samples.detach()
        
        vol_loss = torch.tensor(0.0, requires_grad=True)
        for subj_id, tgt_id, _ in supp_rels:
            if tgt_id in self.arch_elements:
                continue
            else:
                contact_surf = optim_contact_surf[self.id2idx[subj_id]] # (4, 3)
                supp_surf = optim_supp_surf[self.id2idx[subj_id]]
                supp_surf_ctr = supp_surf.mean(0).view(1, 1, 3) # (1, 1, 3)
                supp_two_bound = supp_surf[1:3] - supp_surf[:2] # (2,3)
                supp_bound_lens = torch.norm(supp_two_bound, dim=-1, keepdim=True) # (2,1)
                axes = (supp_two_bound / supp_bound_lens).view(2, 1, 3) # (2,1,3)
                contact_proj_axes = torch.abs((axes * (contact_surf.expand(2, -1, -1) - supp_surf_ctr)).sum(axis=-1)) # (2,4)
                subj_vol_loss = torch.relu(contact_proj_axes - supp_bound_lens / 2).mean() # (2,4)
                
                supp_vec = optim_supp_vec[self.id2idx[subj_id]]
                contact_surf_ctr = contact_surf.mean(0) # (3,)
                contact_surf_diag_len = torch.norm((contact_surf[1:3] - contact_surf[:2]).sum(0))
                samples = optim_supp_samples[self.id2idx[subj_id]]
                sample_offsets = torch.norm(samples - contact_surf_ctr[None, ...], dim=1)
                thres = max(contact_surf_diag_len / 2, sample_offsets.min()+1e-3)
                samples_idx = torch.nonzero(torch.norm(samples - contact_surf_ctr[None, ...], dim=1) <= thres)
                samples_idx = 0 if samples_idx.numel() == 0 else samples_idx
                supp_vol_height = torch.min(optim_obbs['supp_dists'][self.id2idx[subj_id]][samples_idx])
                supp_vol_ctr = (supp_surf.mean(0) + supp_vec * supp_vol_height / 2).view(1, 3)
                subj_corners = optim_corners[self.id2idx[subj_id]]
                subj_proj_swept = torch.abs((supp_vec * (subj_corners - supp_vol_ctr)).sum(axis=-1))
                subj_vol_loss += torch.relu(subj_proj_swept - supp_vol_height / 2).mean()
            
            vol_loss = vol_loss + subj_vol_loss
            
        return vol_loss
            

    # def semantic_orient_loss(self, optim_rot, pre_rot, optim_obbs, pre_obbs, supp_rels):
    #     FLOOR_SUPP = torch.FloatTensor([[0, 1, 0]])
    #     WALL_SUPP = torch.FloatTensor([[0, 0, 1]])
        
    #     optim_front = (optim_rot @ optim_obbs["sem_front"][..., None]).squeeze(2)
    #     optim_supp_vec = (optim_rot @ optim_obbs["supp_vec"][..., None]).squeeze(2)
    #     pre_front = (pre_rot @ pre_obbs["sem_front"][..., None]).squeeze(2)
    #     pre_supp_vec = (optim_rot @ pre_obbs["supp_vec"][..., None]).squeeze(2)
        
    #     supp_vec_ind = [self.id2idx[tgt_id] for _, tgt_id, _ in supp_rels]
    #     optim_supp_vec = torch.cat([optim_supp_vec, FLOOR_SUPP, WALL_SUPP], dim=0)[supp_vec_ind]
    #     pre_supp_vec = torch.cat([pre_supp_vec, FLOOR_SUPP, WALL_SUPP], dim=0)[supp_vec_ind]
        
    #     optim_front_supp_vec_proj = (optim_front * optim_supp_vec).sum(-1, keepdim=True) * optim_front
    #     optim_front_supp_surf_proj = optim_front - optim_front_supp_vec_proj
    #     optim_front_supp_surf_proj = optim_front_supp_surf_proj / torch.norm(optim_front_supp_surf_proj, dim=-1, keepdim=True)
        
    #     pre_front_supp_vec_proj = (pre_front * optim_supp_vec).sum(-1, keepdim=True) * pre_front
    #     pre_front_supp_surf_proj = pre_front - pre_front_supp_vec_proj
    #     pre_front_supp_surf_proj = pre_front_supp_surf_proj / torch.norm(pre_front_supp_surf_proj, dim=-1, keepdim=True)
        
    #     sem_ori_loss = torch.norm(optim_front_supp_surf_proj - pre_front_supp_surf_proj, dim=-1).sum()

    #     return sem_ori_loss


    def semantic_orient_loss(self, optim_rot, pre_rot):
        default_front = torch.tensor([0, 0, 1]).float()
        XZ_PLANE = torch.tensor([1, 0, 1]).float()
        
        optim_front = optim_rot @ default_front
        optim_front_xz = optim_front * XZ_PLANE
        optim_front_xz = optim_front_xz / torch.norm(optim_front_xz, dim=-1, keepdim=True)
        pre_front = pre_rot @ default_front
        pre_front_xz = pre_front * XZ_PLANE
        pre_front_xz = pre_front_xz / torch.norm(pre_front_xz, dim=-1, keepdim=True)
        sem_ori_loss = torch.norm(optim_front_xz - pre_front_xz, dim=-1).sum()

        return sem_ori_loss


    def relative_location_loss(self, optim_obbs, pre_obbs):
        optim_scale = optim_obbs['scale'].detach()
        optim_translation = optim_obbs['translation']
        optim_rot = optim_obbs['rotation'].detach()

        optim_ctr = torch.einsum("bij,bkj->bi", optim_rot.detach(), (optim_scale * optim_obbs['center']).unsqueeze(1)) + optim_translation # (batch, 3)
        pre_ctr = torch.einsum("bij,bkj->bi", pre_obbs["rotation"], (pre_obbs["scale"] * pre_obbs['center']).unsqueeze(1)) + pre_obbs["translation"] # (batch, 3)
        
        rel_loss = torch.tensor(0.0, requires_grad=True)
        # for tgt_id, subj_id, _ in relative_rels:
        for tgt_id in self.objects:
            for subj_id in self.objects:
                if tgt_id == subj_id: continue
                # single_loss = torch.norm((optim_ctr[self.id2idx[tgt_id]]-optim_ctr[self.id2idx[subj_id]]+pre_ctr[self.id2idx[subj_id]]-optim_ctr[self.id2idx[subj_id]]) - (pre_ctr[self.id2idx[tgt_id]]-pre_ctr[self.id2idx[subj_id]]))
                single_loss = torch.norm((optim_ctr[self.id2idx[tgt_id]]-optim_ctr[self.id2idx[subj_id]]) - (pre_ctr[self.id2idx[tgt_id]]-pre_ctr[self.id2idx[subj_id]]))
                rel_loss = rel_loss + single_loss
        
        # return rel_loss / len(relative_rels)
        return rel_loss / (len(self.objects)*(len(self.objects)-1))


    def absolute_location_loss(self, optim_obbs, pre_obbs):
        optim_scale = optim_obbs['scale'].detach()
        optim_translation = optim_obbs['translation']
        optim_rot = optim_obbs['rotation'].detach()
        
        optim_ctr = torch.einsum("bij,bkj->bi", optim_rot.detach(), (optim_scale * optim_obbs['center']).unsqueeze(1)) + optim_translation # (batch, 3)
        pre_ctr = torch.einsum("bij,bkj->bi", pre_obbs["rotation"], (pre_obbs["scale"] * pre_obbs['center']).unsqueeze(1)) + pre_obbs["translation"] # (batch, 3)
        abs_loss = torch.norm(optim_ctr - pre_ctr, dim=-1).mean()
        
        return abs_loss


    def collision_loss(self, optim_obbs, bdb3d_b=None, toleration_dis=0.):
        """
        Test if two bdb3d has collision with Separating Axis Theorem.
        If toleration_dis is positive, consider bdb3ds colliding with each other with specified distance as separate.

        Parameters
        ----------
        bdb3d_a: bdb3d dict
        bdb3d_b: bdb3d dict
        toleration_dis: distance of toleration

        Returns
        -------
        labels:
            0: no collision
            1: has collision
        collision_err_allaxes: collision errors for backpropagation
        """
        axes = optim_obbs['rotation'].permute(2, 0, 1).detach() # (3,n,3)
        # assert not isinstance(axes[0], torch.Tensor) or bdb3d_b is None
        # bdb3d_a = expand_bdb3d(bdb3d_a, - toleration_dis / 2)
        # bdb3d_a_corners = bdb3d_corners(bdb3d_a) # (n,8,3) corners
        optim_scale = optim_obbs['scale'].detach() # (n,3)
        optim_translation = optim_obbs['translation'] # (n,3)
        optim_rot = optim_obbs['rotation'].detach() # (n,8,3)
        bdb3d_a_corners = torch.einsum("bij,bnj->bni", optim_rot, (optim_scale.unsqueeze(1) * optim_obbs['corners'])) + optim_translation.unsqueeze(1) # (n,8,3) corners
        n_bdb3d = len(bdb3d_a_corners)

        shadow_collision_allaxes = None # if the shadows projected on all three axes have overlaps
        # shadow_a_in_b_allaxes = None # if the shadows of a projected on all three axes of b is contained by b
        # shadow_b_in_a_allaxes = None
        collision_err_allaxes = None
        for axis in axes:
            axis = axis.expand(n_bdb3d, -1, -1)[..., None, :].transpose(0, 1)  # (n,n,1,3) axis
            bdb3d_a_corners = bdb3d_a_corners.expand(n_bdb3d, -1, -1, -1) # (n,n,8,3) corners
            bdb3d_corners_proj2a = (axis * bdb3d_a_corners).sum(axis=-1) # (n,n,8), 8 corners projection on axis
            shadow_b = torch.stack([bdb3d_corners_proj2a.min(axis=-1)[0], bdb3d_corners_proj2a.max(axis=-1)[0]]) # (2,n,n), 2nd n means n axis, 3rd n means n obbs
            shadow_a = shadow_b[:, range(n_bdb3d), range(n_bdb3d)] # (2,n), n obbs projected on their own axis
            shadow_a = shadow_a[..., None].expand(-1, -1, n_bdb3d) # (2,n,n), 2nd n means n axis, 3rd means repeated obbs

            a_in_b = [(shadow_b[0] <= shadow_a_end) & (shadow_a_end <= shadow_b[1]) for shadow_a_end in shadow_a] # whether b contains a, test max and min ends seperately
            b_in_a = [(shadow_a[0] <= shadow_b_end) & (shadow_b_end <= shadow_a[1]) for shadow_b_end in shadow_b] # whether a contains b, test max and min ends seperately

            # shadow_a_in_b = a_in_b[0] & a_in_b[1] # if the shadows of a is contained by b
            # shadow_a_in_b_allaxes = shadow_a_in_b if shadow_a_in_b_allaxes is None else shadow_a_in_b_allaxes & shadow_a_in_b
            # shadow_b_in_a = b_in_a[0] & b_in_a[1]
            # shadow_b_in_a_allaxes = shadow_b_in_a if shadow_b_in_a_allaxes is None else shadow_b_in_a_allaxes & shadow_b_in_a
            shadow_collision = a_in_b[0] | a_in_b[1] | b_in_a[0] | b_in_a[1] # if shadows have overlaps
            shadow_collision_allaxes = shadow_collision if shadow_collision_allaxes is None else shadow_collision_allaxes & shadow_collision

            collision_err = torch.min(torch.abs(shadow_a[1] - shadow_b[0]), torch.abs(shadow_a[0] - shadow_b[1]))
            collision_err_allaxes = collision_err if collision_err_allaxes is None else (collision_err_allaxes + collision_err)
            # touch_err = collision_err.clone()
            # touch_err[shadow_collision] = 0.
            # touch_err_allaxes = touch_err if touch_err_allaxes is None else (touch_err_allaxes + touch_err)

        labels = torch.zeros_like(shadow_collision_allaxes, dtype=torch.uint8, device=shadow_collision_allaxes.device)
        labels[shadow_collision_allaxes & shadow_collision_allaxes.T] = 1
        # labels[shadow_a_in_b_allaxes & shadow_a_in_b_allaxes.T] = 2
        # labels[shadow_b_in_a_allaxes & shadow_b_in_a_allaxes.T] = 3
        labels[range(n_bdb3d), range(n_bdb3d)] = 0
        num_collision = labels.sum()

        collision_err_allaxes = collision_err_allaxes + collision_err_allaxes.T
        collision_err_allaxes[torch.logical_not(labels)] = 0.
        collision_err_allaxes[range(n_bdb3d), range(n_bdb3d)] = 0.
        
        # collision_loss = collision_err_allaxes.sum() / num_collision if num_collision else collision_err_allaxes.mean()
        collision_loss = collision_err_allaxes.mean()

        # touch_err_allaxes = touch_err_allaxes + touch_err_allaxes.T
        # touch_err_allaxes[labels > 0] = 0.
        # touch_err_allaxes[range(n_bdb3d), range(n_bdb3d)] = 0.

        return labels, collision_loss


    def rotation_loss(self, optim_obbs, pre_obbs):
        optim_rot = euler_angles_to_matrix(optim_obbs["euler"], "XYZ") # (batch, 3, 3)
        
        align_loss = self.support_align_loss(optim_rot, optim_obbs, self.RELATION["support"]) 
        
        ori_loss = self.semantic_orient_loss(optim_rot, pre_obbs["rotation"])
        # ori_loss = semantic_orient_loss(optim_rot, pre_obbs["rotation"], optim_obbs, pre_obbs, RELATION['support'])
        
        return 3*align_loss + ori_loss


    def translation_loss(self, optim_obbs, pre_obbs):
        place_loss = self.place_on_loss(optim_obbs, self.RELATION["support"]) 
        
        rel_loss = self.relative_location_loss(optim_obbs, pre_obbs)

        return 5*place_loss + rel_loss 


    def scale_loss(self, optim_obbs):
        # space_loss = self.support_space_loss(optim_obbs, self.RELATION["support"])
        space_loss = self.support_volume_loss(optim_obbs, self.RELATION["support"])
        
        # place_loss = self.place_on_loss(optim_obbs, self.RELATION["support"]) 
        
        return space_loss
    
    
    def robust_loss(self, optim_obbs):
        place_loss = self.place_on_loss(optim_obbs, self.RELATION["support"]) 
        
        # space_loss = self.support_volume_loss(optim_obbs, self.RELATION["support"])
        
        _, col_loss = self.collision_loss(optim_obbs)
        
        # adhere_loss = self.adherence_loss(optim_obbs)
        
        return 5*place_loss + col_loss
        
    
    def total_loss(self, optim_obbs, pre_obbs):
        rot_loss = self.rotation_loss(optim_obbs, pre_obbs)
        trans_loss = self.translation_loss(optim_obbs, pre_obbs)
        scale_loss = self.scale_loss(optim_obbs)
        robust_loss = self.robust_loss(optim_obbs)
        
        return rot_loss + trans_loss + scale_loss + robust_loss
    
    
    @torch.no_grad()
    def compute_errors(self, optim_obbs):
        optim_scale = optim_obbs['scale'].detach()
        optim_rot = optim_obbs['rotation'].detach()
        optim_translation = optim_obbs['translation'].detach()
        
        supp_vec, contact_vec = optim_obbs["supp_vec"], optim_obbs["contact_vec"]
        optim_supp_vec = (optim_rot[self.paired_supp_idx].detach() @ supp_vec[..., None]).squeeze(2)
        optim_contact_vec = (optim_rot @ contact_vec[..., None]).squeeze(2)
        
        for subj_id, tgt_id, _ in self.RELATION["support"]:
            if tgt_id in self.arch_elements:
                if tgt_id not in self.archs:
                    continue
                subj_vec = optim_contact_vec[self.id2idx[subj_id]]
                tgt_vec = self.archs_export[tgt_id]["supp_vec"]
            else:
                subj_vec = optim_contact_vec[self.id2idx[subj_id]]
                tgt_vec = optim_supp_vec[self.id2idx[subj_id]]
            vec_diff = torch.acos(torch.clamp(subj_vec @ tgt_vec, -1, 1)).item()
            supp_re = vec_diff / np.pi * 180
            self.scenespec.update_support_re(subj_id, tgt_id, supp_re)

        optim_contact_ctr = optim_obbs['contact_surf'].mean(dim=1) # (batch, 3)
        optim_contact_ctr = torch.einsum("bij,bkj->bi", optim_rot, (optim_scale * optim_contact_ctr).unsqueeze(1)) + optim_translation # (batch, 3)
        optim_supp_surf = optim_obbs['supp_surf'] # (n, 4, 3)
        optim_supp_surf = torch.einsum("bij,bnj->bni", optim_rot[self.paired_supp_idx], (optim_scale[self.paired_supp_idx].unsqueeze(1) * optim_supp_surf)) + optim_translation[self.paired_supp_idx].unsqueeze(1) # (n, 4, 3)
        optim_supp_vec = optim_obbs['supp_vec'] # (n, 3)
        optim_supp_vec = (optim_rot[self.paired_supp_idx] @ optim_supp_vec[..., None]).squeeze(2)

        for subj_id, tgt_id, _ in self.RELATION["support"]:
            if tgt_id in self.arch_elements:
                if tgt_id not in self.archs:
                    continue
                supp_te = torch.norm(((optim_contact_ctr[self.id2idx[subj_id]] - self.archs_export[tgt_id]["corners"][0]) * self.archs_export[tgt_id]["supp_vec"]).sum()).item()
            else:
                supp_te = torch.norm(((optim_contact_ctr[self.id2idx[subj_id]] - optim_supp_surf[self.id2idx[subj_id],0,:]) * optim_supp_vec[self.id2idx[subj_id]]).sum()).item()
            self.scenespec.update_support_te(subj_id, tgt_id, supp_te)
    
    
    def optimize(self, pre_obbs, steps=100, visual=False, lr=0.01, momentum=0.9):
        # initialize optimization
        optim_obbs = {k: v.detach().clone() for k, v in pre_obbs.items()}
        optim_obbs["euler"] = matrix_to_euler_angles(optim_obbs["rotation"], "XYZ")
        
        self.compute_errors(optim_obbs)
        self.scenespec.save_as_json(self.output_dir+"/sg_optm_pre.json")
        
        # for k, v in optim_obbs.items():
        #     if k in ["scale", "euler", "translation"]:
        #         v.requires_grad = True
        # optimizer = torch.optim.SGD([
        #     {'params': optim_obbs["scale"], 'lr': lr/10},
        #     {'params': optim_obbs["translation"], 'lr': lr},
        #     {'params': optim_obbs["euler"], 'lr': lr}
        # ], lr=lr, momentum=momentum)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        # with torch.enable_grad():
        #     for step in range(300):
        #         optimizer.zero_grad()
        #         loss = self.total_loss(optim_obbs, pre_obbs)
        #         loss.backward()
        #         optimizer.step()
        #         scheduler.step()
        
        
        optim_obbs["euler"].requires_grad = True
        optimizer1 = torch.optim.SGD([v for k,v in optim_obbs.items() if k in ["euler"]], lr=lr, momentum=momentum)
        scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=50, gamma=0.1)
        with torch.enable_grad():
            for step in range(300):
                optimizer1.zero_grad()
                rot_loss = self.rotation_loss(optim_obbs, pre_obbs)
                rot_loss.backward()
                optimizer1.step()
                scheduler1.step()
        
        optim_obbs["euler"].requires_grad = False
        for k, v in optim_obbs.items():
            if k in ["scale", "translation"]:
                v.requires_grad = True
        optim_obbs["rotation"] = euler_angles_to_matrix(optim_obbs["euler"], "XYZ")
        
        if visual:
            inter_obbs = {k: v.detach().numpy() for k, v in optim_obbs.items() if k in self.OBB_KEYS}
            inter_obbs = dict_of_array_to_list_of_dict(inter_obbs)
            for i, i_obb in enumerate(inter_obbs):
                self.obbs[i].obb = i_obb
                self.obbs[i].compose_transform()
                view_trans_nmz = self.obbs[i].trs_mat @ self.obbs[i].norm_mat
                self.scenespec.objects[self.idx2id[i]]["view_trans"] = view_trans_nmz.tolist()
            self.export_layout_mesh(self.output_dir+'/scene/layout_after_rot.ply')
        self.compute_errors(optim_obbs)
        self.scenespec.save_as_json(self.output_dir+"/sg_optm_after_rot.json")

        optimizer2 = torch.optim.SGD([v for k,v in optim_obbs.items() if k in ["translation"]], lr=lr, momentum=momentum)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=50, gamma=0.1)
        with torch.enable_grad():
            for step in range(200):
                optimizer2.zero_grad()
                trans_loss = self.translation_loss(optim_obbs, pre_obbs)
                trans_loss.backward()
                optimizer2.step()
                scheduler2.step()
        
        if visual:
            inter_obbs = {k: v.detach().numpy() for k, v in optim_obbs.items() if k in self.OBB_KEYS}
            inter_obbs = dict_of_array_to_list_of_dict(inter_obbs)
            for i, i_obb in enumerate(inter_obbs):
                self.obbs[i].obb = i_obb
                self.obbs[i].compose_transform()
                view_trans_nmz = self.obbs[i].trs_mat @ self.obbs[i].norm_mat
                self.scenespec.objects[self.idx2id[i]]["view_trans"] = view_trans_nmz.tolist()
            self.export_layout_mesh(self.output_dir+'/scene/layout_after_trans.ply')
        self.compute_errors(optim_obbs)
        self.scenespec.save_as_json(self.output_dir+"/sg_optm_after_trans.json")
        
        optimizer3 = torch.optim.SGD([
            {'params': optim_obbs["scale"], 'lr': lr/100},
            {'params': optim_obbs["translation"], 'lr': lr/10}
        ], lr=lr/10, momentum=momentum)
        scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=100, gamma=0.1)
        with torch.enable_grad():
            for step in range(200):
                optimizer3.zero_grad()
                loss = self.scale_loss(optim_obbs)
                loss.backward()
                optimizer3.step()
                scheduler3.step()
        
        if visual:
            inter_obbs = {k: v.detach().numpy() for k, v in optim_obbs.items() if k in self.OBB_KEYS}
            inter_obbs = dict_of_array_to_list_of_dict(inter_obbs)
            for i, i_obb in enumerate(inter_obbs):
                self.obbs[i].obb = i_obb
                self.obbs[i].compose_transform()
                view_trans_nmz = self.obbs[i].trs_mat @ self.obbs[i].norm_mat
                self.scenespec.objects[self.idx2id[i]]["view_trans"] = view_trans_nmz.tolist()
            self.export_layout_mesh(self.output_dir+'/scene/layout_after_scale.ply')
        self.compute_errors(optim_obbs)
        self.scenespec.save_as_json(self.output_dir+"/sg_optm_after_scale.json")
        
        optimizer4 = torch.optim.SGD([v for k,v in optim_obbs.items() if k in ["translation"]], lr=lr/10, momentum=momentum)
        scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=50, gamma=0.1)
        with torch.enable_grad():
            for step in range(200):
                optimizer4.zero_grad()
                loss = self.robust_loss(optim_obbs)
                loss.backward()
                optimizer4.step()
                scheduler4.step()
        
        if visual:
            inter_obbs = {k: v.detach().numpy() for k, v in optim_obbs.items() if k in self.OBB_KEYS}
            inter_obbs = dict_of_array_to_list_of_dict(inter_obbs)
            for i, i_obb in enumerate(inter_obbs):
                self.obbs[i].obb = i_obb
                self.obbs[i].compose_transform()
                view_trans_nmz = self.obbs[i].trs_mat @ self.obbs[i].norm_mat
                self.scenespec.objects[self.idx2id[i]]["view_trans"] = view_trans_nmz.tolist()
            self.export_layout_mesh(self.output_dir+'/scene/layout_after_robust.ply')
        self.compute_errors(optim_obbs)
        self.scenespec.save_as_json(self.output_dir+"/sg_optm_after_robust.json")
        
        final_obbs = {k: v.detach().numpy() for k, v in optim_obbs.items() if k in self.OBB_KEYS}
        final_obbs = dict_of_array_to_list_of_dict(final_obbs)
        for i, f_obb in enumerate(final_obbs):
            self.obbs[i].obb = f_obb
            self.obbs[i].compose_transform()
            self.obbs[i].export_json(os.path.join(self.output_dir, f'cad/{self.idx2id[i]}.obb.optm.json'))
            view_trans_nmz = self.obbs[i].trs_mat @ self.obbs[i].norm_mat
            self.scenespec.objects[self.idx2id[i]]["view_trans"] = view_trans_nmz.tolist()
        self.compute_errors(optim_obbs)
        self.scenespec.save_as_json(self.output_dir+"/sg_optm.json")
    
    
    def export_layout_mesh(self, save_path):
        meshes = []
        for obb in self.obbs:
            obb.compose_transform()
            meshes.append(obb.export_obb_mesh(color=np.random.rand(1,3)*255))
        for _, arch in self.archs.items():
            meshes.append(arch.export_arch_mesh_2d())
        layout_mesh = sum(meshes)
        layout_mesh.export(save_path)
    
    
    def export_scene_mesh(self, save_path, **kwargs):
        meshes = []
        for i, obb in enumerate(self.obbs):
            cad_pick = self.objects[self.idx2id[i]]['cad_pick']
            cad_entry = self.objects[self.idx2id[i]]['retrieval'][cad_pick]
            mesh = load_mesh(cad_entry, normalize=False, base_color=(np.random.rand(3)*255).tolist()+[255])
            mesh.apply_transform(obb.norm_mat)
            mesh.apply_transform(obb.trs_mat)
            meshes.append(mesh)
        for _, arch in self.archs.items():
            meshes.append(arch.export_arch_mesh_2d())
        scene_mesh = sum(meshes)
        scene_mesh.export(save_path)
        
        # im = render_with_view_T(scene_mesh, np.eye(4), img_size=kwargs['img_size'], data_source=kwargs['data_source'], scene_name=kwargs['scene_name'])
        # Image.fromarray(im).save(save_path.replace('.ply', '.png'))


if __name__ == '__main__':
    query_name = 'scene00007'
    output_dir = f'./output/gt/wss/{query_name}'
    
    scenespec = SceneSpec(json.load(open(output_dir+"/sg_pose.json")))
    
    optimizer = LayoutOptimizer(output_dir, scenespec)
    optimizer.export_layout_mesh(output_dir+'/layout_init.ply')
    optimizer.optimize(optimizer.to_optimization(), visual=True)
    optimizer.export_scene_mesh(output_dir+'/scene_f.ply')