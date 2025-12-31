import os
import hydra
import json
from diorama.vis_agent import VisionAgent
from diorama.vlm_agent import VLMAgent
from diorama.scenespec import SceneSpec


@hydra.main(version_base=None, config_path="config", config_name="cfg")
def main(cfg):
    if cfg.img_path is None:
        scenes = sorted(os.listdir(cfg.data.path))
        scenes = [os.path.join(cfg.data.path, scene) for scene in scenes]
        cfg.exp_path = os.path.join(cfg.exp_path, cfg.data.source)
    else:
        scenes = [cfg.img_path]

    vis_agent = VisionAgent(cfg)
    vlm_agent = VLMAgent.create_from_cfg(cfg)
    
    for scene_or_path in scenes:
        vis_agent.read_image(scene_or_path)
        vlm_agent.read_image(scene_or_path)
        output_dir = vis_agent.output_dir

        if cfg.load_depth_model:
            vis_agent.estimate_depth(save_depth=True, save_pcd=True)
        
        if cfg.vlm.sun:
            classes = vlm_agent.understand_scene()
        else:
            sun_json_path = f'{output_dir}/sun.json'
            classes = json.load(open(sun_json_path))["classes"]

        if cfg.load_perception_model:
            vis_agent.detect_objects_w_owlv2(classes)
            vis_agent.segment_objects()
            vis_agent.parse_detects_and_segments()
            aug_image, obj_list = vis_agent.augment_image_w_marks()
            vis_agent.segment_object_crops_and_pcd()
            scenespec = vlm_agent.generate_scene_graph_from_augment(aug_image, obj_list)
        else:
            scenespec = None

        if cfg.load_retrieval_model:
            if scenespec is None:
                sg_json_path = f'{output_dir}/sg.json'
                scenespec = SceneSpec.create_from_json(sg_json_path)
            vis_agent.retrieve_shapes(scenespec)
        
        if cfg.load_pose_model:
            sg_json_path = f'{output_dir}/sg_ret.json'
            scenespec = SceneSpec.create_from_json(sg_json_path)
            vis_agent.estimate_poses(scenespec)
            
        if cfg.run_optimization:
            sg_json_path = f'{output_dir}/sg_ret.json'
            scenespec = SceneSpec.create_from_json(sg_json_path)
            vis_agent.optimize_layout(scenespec)


if __name__ == '__main__':
    main()