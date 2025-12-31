import json
import os
import shutil

import cv2
import faiss
import h5py
import hydra
import numpy as np
import open3d as o3d
import supervision as sv
import torch
import torchvision
from PIL import Image

from diorama.scenespec import SceneSpec
from diorama.model.layout_optim import LayoutOptimizer
from diorama.utils.cad_util import load_mesh_and_save_obb
# from diorama.utils.depth_util import back_project_depth_to_points
from diorama.utils.img_util import resize_padding, multiple_instances_suppression
from diorama.utils.som_util import Visualizer


class VisionAgent:

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.output_dir = cfg.exp_path
        self._classes = self._objects = None
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._init_models(cfg)


    def _init_models(self, cfg):
        models = {}

        if cfg.load_depth_model == "metric3d":
            models["depth"] = hydra.utils.instantiate(cfg.depth_model.metric3d).to(self.device).eval()
            print("Metric3D loaded...")

        if cfg.load_perception_model:
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
            det_processor = Owlv2Processor.from_pretrained(cfg.perception_model.owlv2_processor)
            det_model = Owlv2ForObjectDetection.from_pretrained(cfg.perception_model.owlv2_ckpt_path).to(self.device).eval()
            models["detector"] = (det_model, det_processor)
            print("Detector loaded...")

            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            sam2 = build_sam2(cfg.perception_model.sam2_cfg, cfg.perception_model.sam2_ckpt_path, device=self.device)
            models["sam2"] = SAM2ImagePredictor(sam2)
            print("SAM2 loaded...")

        if cfg.load_retrieval_model == "duoduoclip":
            models["retriever"] = hydra.utils.instantiate(cfg.retrieval_model.duoduoclip).to(self.device).eval()
            print("DuoduoCLIP loaded...")
            self._load_duoduoclip_embs()
        
        if cfg.load_pose_model == "gigazsp":
            models["pose"] = hydra.utils.instantiate(cfg.pose_model.gigazsp)
            print("GigaZSP loaded...")
        
        
        self.models = models
    
    
    def _load_duoduoclip_embs(self):
        non_supp_feats, supp_feats = [], []
        non_supp_cads, supp_cads = [], []
        if self.cfg.cad_pool.wss.is_required:
            wss_names = np.array([n.strip() for n in open(self.cfg.cad_pool.wss.metadata)])
            wss_h5 = h5py.File(self.cfg.cad_pool.wss.ddclip_path, 'r')
            wss_feats = wss_h5["shape_feat"][:].astype(np.float32)
            non_supp_feats.append(wss_feats)
            non_supp_cads.append(wss_names)
            supp_feats.append(wss_feats)
            supp_cads.append(wss_names)
            wss_h5.close()
            print("WSS clean CAD models loaded...")
        
        if self.cfg.cad_pool.fpmodels.is_required:
            fp_names = np.array([n.strip() for n in open(self.cfg.cad_pool.fpmodels.metadata)])
            fp_h5 = h5py.File(self.cfg.cad_pool.fpmodels.ddclip_path, 'r')
            fp_feats = fp_h5["shape_feat"][:].astype(np.float32)
            non_supp_feats.append(fp_feats)
            non_supp_cads.append(fp_names)
            supp_feats.append(fp_feats)
            supp_cads.append(fp_names)
            fp_h5.close()
            print("HSSD clean CAD models loaded...")
        
        if self.cfg.cad_pool.objaverse_lvis.is_required:
            objlvis_names = np.array([n.strip() for n in open(self.cfg.cad_pool.objaverse_lvis.metadata)])
            objlvis_h5 = h5py.File(self.cfg.cad_pool.objaverse_lvis.ddclip_path, 'r')
            objlvis_feats = objlvis_h5["shape_feat"][:].astype(np.float32)
            non_supp_feats.append(objlvis_feats)
            non_supp_cads.append(objlvis_names)
            objlvis_h5.close()
            print("Objaverse-LVIS clean CAD models loaded...")
        
        self.non_supp_feats = np.concatenate(non_supp_feats, axis=0)
        self.non_supp_faiss = faiss.IndexFlatIP(self.non_supp_feats.shape[1])
        self.non_supp_faiss.add(self.non_supp_feats)
        self.supp_feats = np.concatenate(supp_feats, axis=0)
        self.supp_faiss = faiss.IndexFlatIP(self.supp_feats.shape[1])
        self.supp_faiss.add(self.supp_feats)
        
        self.non_supp_cads = np.concatenate(non_supp_cads, axis=0)
        self.supp_cads = np.concatenate(supp_cads, axis=0)
    
        
    def _set_classes(self, classes):
        self._classes = classes
    
    
    def read_image(self, scene_dir_or_img_path, scene_name=None):
        if os.path.isfile(scene_dir_or_img_path):
            self.scene_dir = os.path.dirname(scene_dir_or_img_path)
            img_path = scene_dir_or_img_path
            self.scene_name = scene_name if scene_name is not None else os.path.splitext(os.path.basename(img_path))[0]
        elif os.path.isdir(scene_dir_or_img_path):
            self.scene_dir = scene_dir_or_img_path
            img_path = os.path.join(scene_dir_or_img_path, "scene.png")
            if not os.path.exists(img_path):
                img_path = img_path.replace(".png", ".jpg")
            self.scene_name = scene_name if scene_name is not None else os.path.basename(scene_dir_or_img_path)
        else:
            raise FileNotFoundError
        # self.output_dir = os.path.join(self.cfg.exp_path, self.scene_name)
        self.output_dir = self.cfg.exp_path
        os.makedirs(self.output_dir, exist_ok=True)

        self.pil_image = Image.open(img_path).convert('RGB')
        self.cv2_image = cv2.imread(img_path) # BGR order
        self.width, self.height = self.pil_image.size
    
    
    @torch.no_grad()
    def detect_objects_w_gdino(self, classes, nms=True, save_det=True):
        self._set_classes(classes)
        
        detections = self.models["detector"].predict_with_classes(
            image=self.cv2_image,
            classes=self._classes,
            box_threshold=self.cfg.perception_model.BOX_THRESHOLD,
            text_threshold=self.cfg.perception_model.TEXT_THRESHOLD
        )
        
        # NMS post process
        if nms:
            self.run_detection_nms(detections)
        self.detections = detections
    
    
    @torch.no_grad()
    def detect_objects_w_owlv2(self, classes, run_nms=True, run_mis=True):
        self._set_classes(classes)
        det_model, inp_processor = self.models["detector"]
        
        texts = [[f"a photo of {c}" for c in self._classes]]
        inputs = inp_processor(text=texts, images=self.pil_image, return_tensors="pt").to("cuda:0")
        
        # forward pass
        outputs = det_model(**inputs)
        net_h, net_w = inputs.pixel_values.shape[-2:]
        # Convert outputs (bounding boxes and class logits) to final bounding boxes and scores
        results = inp_processor.post_process_object_detection(
            outputs=outputs, threshold=self.cfg.perception_model.SCORE_THRESHOLD, target_sizes=torch.Tensor([[net_h, net_w]])
        )
        
        # post-process bboxes to the original image size
        rescale = max((self.height, self.width)) / max((net_h, net_w))
        for result in results:
            result["boxes"][:,[0,2]] = result["boxes"][:,[0,2]] * rescale
            result["boxes"][:,[1,3]] = result["boxes"][:,[1,3]] * rescale
            result["boxes"][result["boxes"] < 0.] = 0.

        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        text = texts[i]
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
            
        xyxy = torchvision.ops.box_convert(boxes=boxes, in_fmt="xyxy", out_fmt="xyxy").cpu().numpy()
        detections = sv.Detections(xyxy=xyxy, confidence=scores.cpu().numpy(), class_id=labels.cpu().numpy())
        
        # postprocess
        if run_nms:
            self.run_detection_nms(detections)
        if run_mis:
            self.run_detection_mis(detections)
        self.detections = detections
            
    
    def run_detection_nms(self, detections):
        print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            self.cfg.perception_model.NMS_THRESHOLD
        ).numpy().tolist()
        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        print(f"After NMS: {len(detections.xyxy)} boxes")
    
    
    def run_detection_mis(self, detections):
        print(f"Before MIS: {len(detections.xyxy)} boxes")
        predictions = np.concatenate([detections.xyxy, detections.confidence[..., None], detections.class_id[..., None]], axis=1)
        mis_keep = multiple_instances_suppression(predictions, self.cfg.perception_model.MIS_THRESHOLD)
        detections.xyxy = detections.xyxy[mis_keep]
        detections.confidence = detections.confidence[mis_keep]
        detections.class_id = detections.class_id[mis_keep]
        print(f"After MIS: {len(detections.xyxy)} boxes")

    
    @torch.no_grad()
    def segment_objects(self):
        # Prompting SAM with detected boxes
        sam2_predictor = self.models["sam2"]
        sam2_predictor.set_image(cv2.cvtColor(self.cv2_image, cv2.COLOR_BGR2RGB))
        sam_masks = []
        for box in self.detections.xyxy:
            masks, scores, logits = sam2_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            sam_masks.append(masks[index])
            
        self.detections.mask = np.array(sam_masks).astype(bool)
    
    
    def visualize_detects_and_segments(self, vis_dir=None, vis_det=False, vis_seg=False):
        if vis_dir is None:
            vis_dir = self.output_dir
        os.makedirs(vis_dir, exist_ok=True)
        
        if vis_det:
            box_annotator = sv.BoxAnnotator(thickness=1)
            label_annotator = sv.LabelAnnotator(text_scale=0.4, text_padding=5)
            vis_labels = [
                f"{self._classes[class_id]} {confidence:0.2f}" 
                for _, _, confidence, class_id, _, _
                in self.detections]
            annotated_image = box_annotator.annotate(scene=self.cv2_image.copy(), detections=self.detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=self.detections, labels=vis_labels)
            cv2.imwrite(vis_dir+"/det_vis.jpg", annotated_image)
        
        if vis_seg:
            mask_annotator = sv.MaskAnnotator()
            annotated_image = mask_annotator.annotate(scene=self.cv2_image.copy(), detections=self.detections)
            cv2.imwrite(vis_dir+"/seg_vis.jpg", annotated_image)
    
    
    def parse_detects_and_segments(self):
        masks = [mask for mask in self.detections.mask]
        sorted_ind_masks = sorted(list(enumerate(masks)), key=lambda x: x[1].sum()) # ascending order, from small to large objects
        
        objects = {}
        for i, (ind, mask) in enumerate(sorted_ind_masks):
            obj_id = i + 1
            objects[obj_id] = {
                "id": obj_id,
                "confidence": self.detections.confidence[ind].item(),
                "class_id": self.detections.class_id[ind].item(),
                "class": self._classes[self.detections.class_id[ind]],
                "bbox": self.detections.xyxy[ind], # xyxy
                "mask": mask
            }
        if objects:
            seg_masks = np.concatenate([objects[i+1]["mask"][None, ...] for i in range(len(objects))], axis=0)
            np.save(self.output_dir+"/seg.npy", seg_masks)
            seg_infos = []
            for obj in objects.values():
                seg_infos.append({
                    "id": obj["id"],
                    "class_id": obj["class_id"],
                    "class": obj["class"],
                })
            json.dump(seg_infos, open(self.output_dir+"/seg.json", "w"), indent=4)
            
        self._objects = objects
    
    
    def run_detect_and_segment(self, classes, vis_det=False, vis_seg=False):
        self.detect_objects_w_owlv2(classes)
        self.segment_objects()
        self.visualize_detects_and_segments(vis_det=vis_det, vis_seg=vis_seg)
        self.parse_detects_and_segments()
        
    
    def run_detect_and_segment_wild(self, classes, scene_name, source="wild", vis_det=False, vis_seg=False):
        self.detect_objects_w_owlv2(classes)
        self.segment_objects()
        self.visualize_detects_and_segments(vis_det=vis_det, vis_seg=vis_seg)
        
        masks = [mask for mask in self.detections.mask]
        sorted_ind_masks = sorted(list(enumerate(masks)), key=lambda x: x[1].sum()) # ascending order, from small to large objects
        
        objects = {}
        for i, (ind, mask) in enumerate(sorted_ind_masks):
            obj_id = i + 1
            objects[obj_id] = {
                "id": obj_id,
                "confidence": self.detections.confidence[ind].item(),
                "class_id": self.detections.class_id[ind].item(),
                "class": self._classes[self.detections.class_id[ind]],
                "bbox": self.detections.xyxy[ind], # xyxy
                "mask": mask
            }
        
        seg_one_mask = np.concatenate([objects[i+1]["mask"][None, ...] for i in range(len(objects))], axis=0).any(axis=0)
        arch_seg = np.load(f"{self.output_dir}/arch/segmentation_image.npy")
        arch_ids = np.unique(arch_seg).tolist()
        if 0 in arch_ids:
            arch_ids.remove(0)
        for arch_id in arch_ids:
            arch_data = np.load(f"{self.output_dir}/arch/arch/archplanes/wall_{arch_id}.npy", allow_pickle=True).item()
            arch_type = arch_data["arch_type"]
            arch_mask = arch_seg == arch_id
            arch_mask = arch_mask & (~seg_one_mask)
            obj_id = len(objects) + 1
            objects[obj_id] = {
                "id": obj_id,
                "confidence": 1,
                "class": arch_type,
                "bbox": sv.mask_to_xyxy(masks=arch_mask[None, ...])[0], # xyxy
                "mask": arch_mask #* obj_id
            }
        
        if objects:
            seg_masks = np.concatenate([objects[i+1]["mask"][None, ...] for i in range(len(objects))], axis=0)
            np.save(self.output_dir+"/seg.npy", seg_masks)
            seg_infos = []
            for obj in objects.values():
                seg_infos.append({
                    "id": obj["id"],
                    # "class_id": obj["class_id"],
                    "class": obj["class"],
                })
            json.dump(seg_infos, open(self.output_dir+"/seg.json", "w"), indent=4)
            
            # # create new detections with merged objects
            # classes = list(set([obj["class"] for obj in objects.values()]))
            # xyxy = np.concatenate([obj["bbox"][None, ...] for obj in objects.values()], axis=0)
            # scores = np.array([obj["confidence"] for obj in objects.values()])
            # labels = np.array([classes.index(obj["class"]) for obj in objects.values()])
            # merged_detections = sv.Detections(xyxy=xyxy, confidence=scores, mask=seg_masks, class_id=labels)
            
            # box_annotator = sv.BoxAnnotator(thickness=1)
            # label_annotator = sv.LabelAnnotator(text_scale=0.4, text_padding=5)
            # mask_annotator = sv.MaskAnnotator()
            # vis_labels = [f"{obj['class']} {obj['confidence']:0.2f}" for obj in objects.values()]
            # annotated_image = mask_annotator.annotate(scene=self.cv2_image.copy(), detections=merged_detections)
            # annotated_image = box_annotator.annotate(scene=annotated_image, detections=merged_detections)
            # annotated_image = label_annotator.annotate(scene=annotated_image, detections=merged_detections, labels=vis_labels)
            # cv2.imwrite(self.output_dir+"/seg_merge_vis.jpg", annotated_image)
            
        self._objects = objects
        
    
    def load_gt_bboxes_and_segments(self):
        gt = json.load(open(os.path.join(self.scene_dir, "gt.json")))
        obj2classes = {}
        for obj_id in gt:
            obj2classes[int(obj_id)] = gt[obj_id]['class']
        
        seg_path = os.path.join(self.scene_dir, "seg_w_arch.png")
        seg_np = np.array(Image.open(seg_path))
        obj_inds = np.unique(seg_np).tolist()
        
        gt_objects = {}
        for obj_idx in obj_inds:
            if obj_idx <= 1: 
                continue
            mask = seg_np == obj_idx
            obj_id = obj_idx - 1
            # filter out objects whose class is unknown or are too small
            if obj_id not in obj2classes or mask.sum() <= 500: 
                continue
            obj_class = obj2classes[obj_id]
            # filter out objects that are highly likely to be truncated a lot
            if obj_class not in ["wall", "floor", "ceiling"]:
                xx_ind, yy_ind = np.nonzero(mask)
                if mask.sum() < self.width*self.height*0.002 and (np.any((yy_ind == 0) | (yy_ind == self.width-1)) or np.any((xx_ind == 0) | (xx_ind == self.height-1))):
                    continue
            gt_objects[obj_id] = {
                "id": obj_id,
                "confidence": 1,
                "class": obj_class,
                "bbox": sv.mask_to_xyxy(masks=mask[None, ...])[0], # xyxy
                "mask": mask
            }

        self._gt_objects = gt_objects
        
    
    def segment_object_crops_and_pcd(self, reload_pcd=False):
        det_dir = os.path.join(self.output_dir, "detection")
        if os.path.exists(det_dir):
            shutil.rmtree(det_dir)
        pcd_dir = os.path.join(self.output_dir, "pcd")
        if os.path.exists(pcd_dir):
            shutil.rmtree(pcd_dir)
        os.makedirs(det_dir, exist_ok=True)
        os.makedirs(pcd_dir, exist_ok=True)
        
        if not hasattr(self, 'pcd') or reload_pcd:
            self.pcd = np.asarray(o3d.io.read_point_cloud(os.path.join(self.output_dir, "pcd.ply")).points).reshape(self.height, self.width, 3)

        objects = self._objects if self._objects is not None else self._gt_objects
        
        for obj_id, obj in objects.items():
            box = obj["bbox"].astype(int)
            mask = obj["mask"].astype(bool)
            
            inst_rgb = self.cv2_image.copy()
            inst_rgb[~mask] = (0, 0, 0)
            inst_rgb_crop = inst_rgb[box[1]:box[3], box[0]:box[2], :]
            obj["masked_crop"] = inst_rgb_crop
            cv2.imwrite(det_dir+f"/rgb.{obj_id}.jpg", inst_rgb)
            cv2.imwrite(det_dir+f"/mask.{obj_id}.png", mask.astype(np.uint8)*255)
            
            inst_points = self.pcd.copy()
            points_mask = cv2.erode(mask.astype(np.uint8), np.ones((3,3), np.uint8), iterations=1).astype(bool)
            if points_mask.sum() == 0:
                points_mask = mask
            inst_points[~points_mask] = (0, 0, 0)
            np.save(pcd_dir+f"/pcd.{obj_id}.npy", inst_points.astype(np.float16))
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(inst_points[points_mask])
            inst_rgb_points = self.cv2_image.copy()
            inst_rgb_points[~mask] = (0, 0, 0)
            inst_points_colors = (inst_rgb_points[points_mask] / 255.0)[:, ::-1] # bgr to rgb
            pcd.colors = o3d.utility.Vector3dVector(inst_points_colors)
            o3d.io.write_point_cloud(pcd_dir+f"/pcd.{obj_id}.ply", pcd)
            
    
    def augment_image_w_marks(self, label_mode='1', alpha=0.1, anno_mode=['Mark', 'Mask'], vis_aug=False):
        visual = Visualizer(self.pil_image)

        mask_map = np.zeros(self.cv2_image.shape[:2], dtype=np.uint8)
        objects = self._objects if self._objects is not None else self._gt_objects
        if not objects:
            print("No objects detected or segmented.")
            return
        for obj_id, obj in objects.items():
            mask = obj["mask"]
            mask = mask & ~mask_map.astype(bool)
            try:
                demo = visual.draw_binary_mask_with_number(mask, text=str(obj_id), label_mode=label_mode, alpha=alpha, anno_mode=anno_mode)
                # assign the mask to the mask_map
                mask_map[mask == 1] = obj_id
            except:
                print(f"Error in drawing mask for object {obj_id}")
                continue
        self.mask_map = mask_map
        im = demo.get_image()
        self.aug_pil_image = Image.fromarray(im)
        self.obj_list = [(obj_id, obj['class']) for obj_id, obj in objects.items()]
        
        if vis_aug:
            self.aug_pil_image.save(self.output_dir+"/aug.jpg")
    
    
    def update_scenespec_w_visuals(self, scenespec):
        if self._objects is not None:
            for obj_id, obj in scenespec.objects.items():
                obj.update(self._objects[obj_id])
            for obj_id, obj in scenespec.arch_elements.items():
                obj.update(self._objects[obj_id])

    
    @torch.autocast(device_type="cuda")
    def retrieve_shapes(self, scenespec: SceneSpec):
        retriever = self.models["retriever"]
        obj_ids = list(scenespec.objects.keys())
        supp_obj_ind = np.array([obj_ids.index(obj_id) for obj_id in scenespec.supp_obj_ids], dtype=int)
        non_supp_obj_ind = np.array([obj_ids.index(obj_id) for obj_id in scenespec.non_supp_obj_ids], dtype=int)
        
        if self.cfg.retrieval_model.use_caption:
            txt_embs = []
            for obj in scenespec.objects.values():
                if isinstance(obj['caption'], list):
                    obj_captions = [f"A photo of {c.lower()}" for c in obj['caption']]
                    obj_txt_emb = retriever.encode_text(obj_captions).mean(0).cpu().numpy()
                else:
                    obj_txt_emb = retriever.encode_text(f"A photo of {obj['caption'].lower()}").cpu().numpy()
                txt_embs.append(obj_txt_emb)
            txt_embs = np.asarray(txt_embs)
        else:
            captions = [f"A photo of {obj['name']}" for obj in scenespec.objects.values()]
            txt_embs = retriever.encode_text(captions).cpu().numpy()
        
        if self._objects is not None:
            img_crops = [Image.fromarray(self._objects[obj_id]["masked_crop"]) for obj_id in obj_ids]
        else:
            imgs = [Image.open(f"{self.output_dir}/detection/rgb.{obj_id}.jpg") for obj_id in obj_ids]
            masks = [Image.open(f"{self.output_dir}/detection/mask.{obj_id}.png") for obj_id in obj_ids]
            img_crops = []
            for im, mask in zip(imgs, masks):
                im_np = np.array(im)
                im_np[np.array(mask)==0] = (255, 255, 255)
                img_crops.append(Image.fromarray(im_np).crop(mask.getbbox()))
            # img_crops = [im.crop(im.getbbox()) for im in imgs]
        queries = [resize_padding(im, 224, "RGB", background_color=(255, 255, 255), padding=12)[0] for im in img_crops]
        img_embs = retriever.encode_image(queries).cpu().numpy()
    
        txt_embs_supp, txt_embs_non_supp = txt_embs[supp_obj_ind], txt_embs[non_supp_obj_ind]
        img_embs_supp, img_embs_non_supp = img_embs[supp_obj_ind], img_embs[non_supp_obj_ind]
        
        if len(non_supp_obj_ind) != 0:
            print("Searching non-supporting objects knn...")
            if self.cfg.retrieval_model.text_only:
                St, It = self.non_supp_faiss.search(txt_embs_non_supp, self.cfg.retrieval_model.num_knn)
                cad_names_non_supp = [self.non_supp_cads[i_knn].tolist() for i_knn in It]
            elif self.cfg.retrieval_model.img_only:
                Si, Ii = self.non_supp_faiss.search(img_embs_non_supp, self.cfg.retrieval_model.num_knn)
                cad_names_non_supp = [self.non_supp_cads[i_knn].tolist() for i_knn in Ii]
            else:
                St, It = self.non_supp_faiss.search(txt_embs_non_supp, self.cfg.retrieval_model.num_txt_knn)
                cad_feats_txt_ret = self.non_supp_feats[It] # (num_query, num_txt_knn, dim)
                cad_img_sim = (cad_feats_txt_ret * img_embs_non_supp[:,None,:]).sum(-1) # (num_query, num_txt_knn)
                cad_img_sim = cad_img_sim + St
                # cad_img_sim[St < self.cfg.retrieval_model.sim_thres] = 0
                cad_ind_non_supp = np.take_along_axis(It, np.argsort(cad_img_sim)[:,::-1], axis=1)
                cad_names_non_supp = [self.non_supp_cads[i_knn].tolist() for i_knn in cad_ind_non_supp]
        else:
            cad_names_non_supp = []
        
        if len(supp_obj_ind) != 0:
            print("Searching clean supporting objects knn...")
            if self.cfg.retrieval_model.text_only:
                St, It = self.supp_faiss.search(txt_embs_supp, self.cfg.retrieval_model.num_knn)
                cad_names_supp = [self.supp_cads[i_knn].tolist() for i_knn in It]
            elif self.cfg.retrieval_model.img_only:
                Si, Ii = self.supp_faiss.search(img_embs_supp, self.cfg.retrieval_model.num_knn)
                cad_names_supp = [self.supp_cads[i_knn].tolist() for i_knn in Ii]
            else:
                St, It_supp = self.supp_faiss.search(txt_embs_supp, self.cfg.retrieval_model.num_txt_knn)
                cad_feats_txt_ret = self.supp_feats[It_supp] # (num_query, num_txt_knn, dim)
                cad_img_sim = (cad_feats_txt_ret * img_embs_supp[:,None,:]).sum(-1) # (num_query, num_txt_knn)
                cad_img_sim = cad_img_sim + St
                # cad_img_sim[St < self.cfg.retrieval_model.sim_thres] = 0
                cad_ind_supp = np.take_along_axis(It_supp, np.argsort(cad_img_sim)[:,::-1], axis=1)
                cad_names_supp = [self.supp_cads[i_knn].tolist() for i_knn in cad_ind_supp]
        else:
            cad_names_supp = []
        
        R = cad_names_non_supp + cad_names_supp
        obj_ind = non_supp_obj_ind.tolist() + supp_obj_ind.tolist()        
        for ind, r in zip(obj_ind, R):
            scenespec.objects[obj_ids[ind]]['retrieval'] = r[:self.cfg.retrieval_model.num_knn]
        scenespec.save_as_json(self.output_dir+f"/{self.cfg.retrieval_model.filename}")
        
    
    def load_gt_shapes(self, scenespec: SceneSpec, gt_filename="gt.json"):
        gt = json.load(open(os.path.join(self.scene_dir, gt_filename)))
        obj_ids = list(scenespec.objects.keys())
        
        for obj_id in obj_ids:
            scenespec.objects[obj_id]['retrieval'] = [gt[str(obj_id)]["model"]]
        scenespec.save_as_json(self.output_dir+f"/{self.cfg.retrieval_model.filename}")
        

    def estimate_poses(self, scenespec: SceneSpec, query_dir=None):
        poser = self.models["pose"]
        cad_dir = f'{self.output_dir}/cad'
        os.makedirs(cad_dir, exist_ok=True)
        if query_dir is None:
            query_dir = self.output_dir
        
        # n_cads = min(len(list(scenespec.objects.values())[0]['retrieval']), 1)
        n_cads = len(list(scenespec.objects.values())[0]['retrieval'])
        for cad_pick in range(n_cads):
            # print("Loading meshes and saving initial OBBs...")
            for obj_id in scenespec.objects:
                cad_entry = scenespec.objects[obj_id]['retrieval'][cad_pick]
                load_mesh_and_save_obb(cad_entry, save_dir=cad_dir, obj_id=f"{obj_id}.{cad_pick}")

            print("Prepare inputs...")
            poser.prepare_inputs(query_dir, scenespec, cad_pick=cad_pick, data_source=self.cfg.data.source, scene_name=self.scene_name)
            scenespec = poser.estimate_poses(self.output_dir)
            
            if self.cfg.pose_model.plot_results:
                poser.viz_correspondence_and_pose(self.output_dir, data_source=self.cfg.data.source, scene_name=self.scene_name, img_size=(self.height, self.width))
        
        scenespec.save_as_json(self.output_dir+"/sg_pose.json")
    
    
    def optimize_layout(self, scenespec: SceneSpec, pose_dir=None, cad_picks=None):
        output_dir = self.output_dir
        if self.cfg.layout_optim.hypo is not None:
            shutil.rmtree(output_dir)
            output_dir = output_dir + f"_{self.cfg.layout_optim.hypo}"
        
        pose_dir = output_dir if pose_dir is None else pose_dir
        cad_dir = os.path.join(output_dir, 'cad')
        os.makedirs(cad_dir, exist_ok=True)
        scene_dir = os.path.join(output_dir, 'scene')
        os.makedirs(scene_dir, exist_ok=True)
        
        optimizer = LayoutOptimizer(scenespec, output_dir, pose_dir, self.cfg.layout_optim.arch_dir, self.cfg.layout_optim.arch_mask_dir, cad_picks=cad_picks)
        optimizer.export_layout_mesh(scene_dir+'/layout_init.ply')
        optimizer.optimize(optimizer.to_optimization(), lr=self.cfg.layout_optim.base_lr, visual=True)
        optimizer.export_layout_mesh(scene_dir+'/layout_f.ply')
        optimizer.export_scene_mesh(scene_dir+'/scene_f.ply', data_source=self.cfg.data.source, scene_name=self.scene_name, img_size=(self.height, self.width))
        
        # scenespec.save_as_json(self.output_dir+"/sg_optm.json")