import os
import json
from collections import Counter
from io import BytesIO
from PIL import Image
import numpy as np
import networkx as nx
import plotly.express as px
from matplotlib import pyplot as plt


graph_color_ref = px.colors.qualitative.Bold + px.colors.qualitative.Prism

def get_color(ref, n_nodes):
    '''
    Function to color the nodes

    Args:
    - ref (list): list of color reference
    - n_nodes (int): number of nodes

    Returns:
    - colors (list): list of colors
    '''
    N = len(ref)
    colors = []
    for i in range(n_nodes):
        colors.append(np.array([[int(i) for i in ref[i%N][4:-1].split(',')]]) / 255.)
    return colors


class SceneSpec:
    # ARCHITECT = ['floor', 'wall', 'ceiling', 'door', 'window']
    ARCHITECT = ['floor', 'wall', 'ceiling']
    
    def __init__(self, scene_graph=None, classes=None, json_path=None):
        self.sg = scene_graph
        self.classes = classes
        self.json_path = json_path
        
        if self.sg is not None:
            self.parse_scene_graph()
    
    
    @classmethod
    def create_from_json(cls, json_path):
        scene_graph = json.load(open(json_path))
        return cls(scene_graph, json_path=json_path)
    
    
    @classmethod
    def create_from_wss_gt(cls, gt, obj_list=None, save_path=None):
        if obj_list is not None:
            obj_list = [o[0] for o in obj_list]
            gt = {obj_id: obj for obj_id, obj in gt.items() if int(obj_id) in obj_list}
        
        objects = [
            {
                "id": int(obj_id),
                "name": obj["name"] if "name" in obj and not obj["class"] else obj["class"],
                "caption": obj["name"] if "name" in obj and not obj["class"] else obj["class"]
            }
            for obj_id, obj in gt.items()
        ]
        support_relations = [
            {
                "subject_id": int(obj_id),
                "type": "placed on",
                "target_id": int(obj["parent_id"])
            }
            for obj_id, obj in gt.items() if "parent_id" in obj and obj["parent_id"] != 0
        ]
        scene_graph = {
            "room_type": "",
            "objects": objects,
            "support_relation": support_relations,
            # "spatial_relation": []
        }
        if save_path is not None:
            with open(save_path, "w") as f:
                json.dump(scene_graph, f, indent=4)
        
        return cls(scene_graph)
    
    def save_as_json(self, save_path=None):
        if save_path is None:
            assert self.json_path is not None
            save_path = self.json_path
        with open(save_path, "w") as f:
            json.dump(self.sg, f, indent=4)
    
    
    def parse_scene_graph(self):
        classes = set()
        objects = {}
        arch_elements = {}
        for obj in self.sg["objects"]:
            cat = obj['name'].lower()
            if "face_direction" in obj and isinstance(obj["face_direction"], list):
                obj["face_direction"] = Counter(obj["face_direction"]).most_common(1)[0][0]
            if cat not in self.ARCHITECT:
                classes.add(cat)
                objects[obj['id']] = obj
            else:
                arch_elements[obj['id']] = obj
        self.classes = list(classes)
        self.objects = objects
        self.arch_elements = arch_elements
        
        support_relations, arch_support_relations = [], []
        supp_obj_ids = set()
        for rel in self.sg["support_relation"]:
            if rel['subject_id'] not in self.objects:
                continue
            if isinstance(rel['target_id'], list):
                rel['type'] = Counter(rel["type"]).most_common(1)[0][0]
                rel['target_id'] = Counter(rel["target_id"]).most_common(1)[0][0]
            if rel['target_id'] not in self.objects and rel['target_id'] not in self.arch_elements:
                continue
            if rel['type'] in ["mounted on", "hung on"] and rel['target_id'] in ["floor"]:
                rel['type'] = "placed on"
                
            support_relations.append((rel['subject_id'], rel['target_id'], rel['type']))
            if rel['target_id'] in self.arch_elements:
                arch_support_relations.append((rel['subject_id'], rel['target_id'], rel['type']))
            supp_obj_ids.add(rel['target_id'])
        
        self.support_relations = support_relations
        self.arch_support_relations = arch_support_relations
        supp_obj_ids = supp_obj_ids.difference(set(self.arch_elements.keys()))
        self.supp_obj_ids = np.array(list(supp_obj_ids))
        self.non_supp_obj_ids = np.setdiff1d(list(self.objects.keys()), self.supp_obj_ids)
    
    
    def add_support_surf(self, subj_id, tgt_id, supp_idx, supp_surf, supp_vec):
        for rel in self.sg["support_relation"]:
            if rel['subject_id'] == subj_id and rel['target_id'] == tgt_id:
                rel['supp_idx'] = int(supp_idx)
                rel['supp_surf'] = supp_surf.tolist()
                rel['supp_vec'] = supp_vec.tolist()
                break
    
    def update_support_re(self, subj_id, tgt_id, supp_re):
        for rel in self.sg["support_relation"]:
            if rel['subject_id'] == subj_id and rel['target_id'] == tgt_id:
                rel['supp_re'] = supp_re
                break
    
    def update_support_te(self, subj_id, tgt_id, supp_te):
        for rel in self.sg["support_relation"]:
            if rel['subject_id'] == subj_id and rel['target_id'] == tgt_id:
                rel['supp_te'] = supp_te
                break
    
    
    def _init_G(self):
        G = nx.DiGraph()
        objects = list(self.objects.items())
        arch_elements = list(self.arch_elements.items())
        G.add_nodes_from(objects)
        G.add_nodes_from(arch_elements)
        relations = [(id1, id2, {'type': relation}) for id1, id2, relation in self.support_relations]
        G.add_edges_from(relations)
        self.G = G
    
    
    def visualize_scene_graph(self, res=500, gt=None, save_dir=None):
        # plt.figure(figsize=(res/100, res/100))
        plt.figure(figsize=(7, 3))

        pos = nx.nx_agraph.graphviz_layout(self.G, prog="twopi", args="")
        node_order = sorted(self.G.nodes())
        
        if gt is not None:
            node_colors = []
            for node_id in node_order:
                color = gt[str(node_id)].get("model_color", [150, 150, 150])
                node_colors.append(np.array(color) / 255.)
        else:
            node_colors = get_color(graph_color_ref, len(self.objects)+len(self.arch_elements))
            
        nx.draw(self.G, pos, node_color=node_colors, nodelist=node_order, node_size=500, edge_color='k', with_labels=False)
        
        node_labels = nx.get_node_attributes(self.G, 'name')
        nx.draw_networkx_labels(self.G, pos, node_labels, font_size=8, font_weight="bold")
        
        # edge_labels = nx.get_edge_attributes(self.G, 'type')
        # nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_size=6, verticalalignment="top")
        
        # buf = BytesIO()
        plt.savefig(os.path.join(save_dir, "sg.png"), transparent=True, format="png", dpi=120)
        # buf.seek(0)
        # img = Image.open(buf)
        # img.save(os.path.join(save_dir, "sg.png"))
        # buf.close()
        plt.clf()
        plt.close()
        

if __name__ == "__main__":
    scene = "scene_37"
    
    sg = json.load(open(f"output/wild/test/{scene}/sg.json"))
    scenespec = SceneSpec(sg)
    scenespec._init_G()
    scenespec.visualize_scene_graph(save_dir=f"output/wild/test/{scene}")