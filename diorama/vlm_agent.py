import os, io, re
import json
import base64
import cv2
from PIL import Image
import numpy as np
from openai import OpenAI, NotGiven

from diorama.scenespec import SceneSpec
from diorama.utils.prompt_util import *


class VLMAgent:
    """
    Communicate with a LVLM to achieve open-world scene understanding, scene graph generation, and object recognition
    """
    MODELS = [
        "gpt-4-vision-preview",
        "gpt-4-turbo",
        "gpt-4o",
        "gpt-4o-2024-08-06"
    ]
    PROMPTS = {
        "sun": SUN_PROMPT,
        "sun_verify": SUN_VERIF_PROMPT,
        "init_sg": SG_INIT_PROMPT,
        "init_sg_aug": SG_INIT_AUG_PROMPT,
        "init_sg_aug_v2": SG_INIT_AUG_PROMPT_V2,
        "arch_adhere": ARCH_ADHERE_PROMPT,
        "object": OBJECT_PROMTP,
        "merge_parts": MERGE_OBJS_AUG_PROMPT
    }

    def __init__(self, model="gpt-4o", output_dir="./output", n_trials=5, cfg=None) -> None:
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.base64_image = None
        self.output_dir = output_dir
        self.n_trials = n_trials
        self.cfg = cfg
    
    
    @classmethod
    def create_from_cfg(cls, cfg):
        return cls(model=cfg.vlm.model, output_dir=cfg.exp_path, n_trials=cfg.vlm.n_trials, cfg=cfg)
        
        
    def encode_base64(self, image_or_path):
        if isinstance(image_or_path, Image.Image):
            buffer = io.BytesIO()
            image_or_path.save(buffer, format="JPEG")  # You can change JPEG to PNG depending on your requirements
            base64_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
        elif isinstance(image_or_path, np.ndarray):
            _, buffer = cv2.imencode('.jpg', image_or_path)  # You can change '.jpg' to '.png'
            base64_img = base64.b64encode(buffer).decode('utf-8')
        elif isinstance(image_or_path, str):
            with open(image_or_path, "rb") as image_file:
                base64_img = base64.b64encode(image_file.read()).decode('utf-8')
        else:
            raise NotImplementedError
        
        return base64_img
    
    
    def read_image(self, scene_dir_or_img_path):
        if os.path.isfile(scene_dir_or_img_path):
            self.scene_dir = os.path.dirname(scene_dir_or_img_path)
            img_path = scene_dir_or_img_path
            self.scene_name = os.path.splitext(os.path.basename(img_path))[0]
        elif os.path.isdir(scene_dir_or_img_path):
            self.scene_dir = scene_dir_or_img_path
            img_path = os.path.join(scene_dir_or_img_path, "scene.png")
            self.scene_name = os.path.basename(scene_dir_or_img_path)
        else:
            raise FileNotFoundError
        # self.output_dir = os.path.join(self.cfg.exp_path, self.scene_name)
        self.output_dir = self.cfg.exp_path
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.base64_image = self.encode_base64(img_path)
    
    
    def load_gt_classes(self):
        gt = json.load(open(os.path.join(self.scene_dir, "gt.json")))
        classes = []
        for obj_id in gt:
            classes.append(gt[obj_id]['class'])
        
        sun_json = {"classes": list(set(classes))}
        with open(self.output_dir+"/sun.json", "w") as f:
            json.dump(sun_json, f, indent=4)
        
        return sun_json["classes"]
    
    
    def understand_scene(self, image=None):
        if image is not None or self.base64_image is None:
            self.base64_image = self.encode_base64(image)
        
        print(f"Chat with {self.model} for scene understanding... Waiting...")
        success = False
        while not success:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text", 
                                    "text": self.PROMPTS["sun"]
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{self.base64_image}",
                                    },
                                },
                            ],
                        }
                    ],
                    response_format={"type": "json_object"} if self.model != 1 else NotGiven()
                )
                sun_json = self.parse_gpt4v_response(response)
                success = True
            except Exception as e:
                print("Retrying...")
        
        classes = sun_json["classes"]
        if self.cfg.vlm.verify_sun:
            classes = self.verify_scene_understanding(sun_json)
            
        with open(self.output_dir+"/sun.json", "w") as f:
            json.dump(sun_json, f, indent=4)
            
        return classes
    
    
    def verify_scene_understanding(self, sun_json):
        classes = sun_json["classes"]
        
        print(f"Chat with {self.model} for verifying scene understanding... Waiting...")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": self.PROMPTS["sun_verify"].format(objects=classes)
                        }
                    ],
                }
            ],
            response_format={"type": "json_object"} if self.model != 1 else NotGiven()
        )
        
        group_json = self.parse_gpt4v_response(response)
        classes_cp = classes.copy()
        for g in group_json["object_groups"]:
            base_object = g["base_object"]
            if base_object not in classes:
                continue
            for o in g["parts"]:
                if o in classes_cp and o not in ["floor", "wall", "ceiling"]:
                    classes_cp.remove(o)
        group_json["merged_classes"] = classes_cp
        sun_json.update(group_json)
            
        return sun_json["merged_classes"]
    
    
    def generate_scene_graph_from_augment(self, aug_image, objects_with_ids) -> SceneSpec:
        base64_aug_image = self.encode_base64(aug_image)
        
        object_list = ''.join([OBJECT_ENTRY.format(obj_id=obj_id, obj_name=obj_name, face_direction=FACE_DIRECTION) for obj_id, obj_name in objects_with_ids])
        obj_ids_wo_arch = [obj_id for obj_id, obj_name in objects_with_ids if obj_name not in ['floor', 'wall', 'ceiling']]
        support_relation_list = ''.join([SUPPORT_RELATION_ENTRY.format(obj_id=obj_id, support_rel=SUPPORT_REL) for obj_id in obj_ids_wo_arch])
        # spatial_relation_list = ''.join([SPATIAL_RELATION_ENTRY.format(obj_id=obj_id, spatial_rel=SPATIAL_REL) for obj_id, _ in objects_with_ids])

        trials = []
        print(f"Chat with {self.model} for scene graph generation... Waiting...")
        while len(trials) < self.n_trials:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text", 
                                    "text": self.PROMPTS["init_sg_aug_v2"].format(objects_with_ids=objects_with_ids,
                                                                                  object_list=object_list,
                                                                                  support_relation_list=support_relation_list,
                                                                                #   spatial_relation_list=spatial_relation_list
                                                                                  )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_aug_image}",
                                    },
                                },
                            ],
                        }
                    ],
                    response_format={"type": "json_object"} if self.model != "gpt-4-vision-preview" else NotGiven()
                )
                sg = self.parse_gpt4v_response(response)
                if "support_relation" not in sg or len(sg["support_relation"]) == 0:
                    print("Retrying...")
                    continue
                
                trials.append(sg)
            
            except Exception as e:
                print("Retrying...")
        
        sg_merge = {k: [] for k in trials[0]}
        for obj_id, obj_name in objects_with_ids:
            objs = [obj for sg in trials for obj in sg["objects"] if obj["id"] == obj_id]
            if objs:
                sg_merge["objects"].append({
                    "id": obj_id,
                    "name": obj_name,
                    "caption": [obj["caption"] for obj in objs],
                    "face_direction": [obj["face_direction"] for obj in objs]
                })
            else:
                sg_merge["objects"].append({"id": obj_id, "name": obj_name})
            rels = [rel for sg in trials for rel in sg["support_relation"] if rel["subject_id"] == obj_id]
            if rels:
                sg_merge["support_relation"].append({
                    "subject_id": obj_id,
                    "type": [rel["type"] for rel in rels],
                    "target_id": [rel["target_id"] for rel in rels]
                })
        
        with open(self.output_dir+"/sg.json", "w") as f:
            json.dump(sg_merge, f, indent=4)
        
        return SceneSpec(scene_graph=sg_merge)
    
    
    def parse_gpt4v_response(self, response):
        raw_out = response.choices[0].message.content
        if self.model == "gpt-4-vision-preview":
            raw_out = re.findall(r'```json\n(.*?)(?=\n```)', raw_out, re.DOTALL)[0]
        try:
            json_out = json.loads(raw_out)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse scene graph response from OpenAI as JSON:\n{raw_out}")
        
        return json_out

