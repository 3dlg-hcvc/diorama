# import json


SUN_PROMPT = """What are objects in the synthetic rendering image? Includes architectural elements such as floor, wall and ceiling if applicable. Ensure that each class is in singular and has no quantifiers.
Return the output in a JSON format according to the following format:
{{
  "classes": [object1, object2, object3, ...]
}}
"""


SUN_VERIF_PROMPT = """A list of objects are given as "{objects}". What objects are functional parts of the other object in the image? 
Return the output in a JSON format according to the following format:
{{
  "object_groups": [
    {{
      "base_object": object name,
      "parts": [part1, part2, ...], (use object name from the reference list)
    }}
  ]
}}
"""


MERGE_OBJS_AUG_PROMPT = """A reference list of labelled items is provided as "{objects_with_ids}" in the format of <object_id, object_name>. What are items belonging to the parts of the same object instance in the image? Exclude objects that are different instances of the same object type.
Return the output in JSON according to the following format:
{{
  "belongings": [
    {{
      "object_ids": a list of object ids (use object_id from the reference list),
      "reason": brief reason of belongings,
    }}
  ]
}}
Return empty JSON if there are no object parts belongings.
"""


SUPPORT_REL = ["placed on", "mounted on"]

SPATIAL_REL = ["on left of", "on right of", "in front of", "behind", "above", "below", "next to", "facing"]

FACE_DIRECTION = ["up", "down", "left", "right", "front", "back"]


OBJECT_ENTRY = """    {{
      "id": {obj_id} (id of object),
      "name": {obj_name} (name of object),
      "caption": brief description of object appearance (excluding relationship),
      “face_direction”: direction of the object (choose from the list {face_direction}),
    }},
"""
SUPPORT_RELATION_ENTRY = """    {{
        "subject_id": {obj_id} (id of object),
        "type": choose from a list of supporting relationships {support_rel},
        "target_id": id of supporting object,
    }},
"""
SPATIAL_RELATION_ENTRY = """    {{
        "subject_id": {obj_id} (id of object),
        "type": choose from a list of spatial relationships {spatial_rel},
        "target_id": id of the other object,
    }},
"""
SG_INIT_AUG_PROMPT_V2 = """Please follow the examples in the Visual Genome dataset and generate a scene graph that best describes the image. A reference list of labelled objects is provided as "{objects_with_ids}" in the format of <object_id, object_name>.
Return the output in JSON according to the following format:
{{
  "room_type": type of room,
  "objects": [
    {object_list}
  ],
  "support_relation": [
    {support_relation_list}
  ]
}}
Relationships are expressed as <subject_id, type, target_id>. Each pair of <subject_id, target_id> only occurs once.
"""


ARCH_ADHERE_PROMPT = """What are archtecture elements that objects placed against to besides the supporting architecture? A reference list of labelled architecture is provided as "{archs_with_ids}" in the format of <arch_id, arch_name>. A reference list of potential object-architecture pair is provided as "{potential_pairs_with_ids}" in the format of <object_id, object_name, arch_id, arch_name>.
Return the output in JSON according to the following format:
{{
  "adherences": [
    {{
      "object_id": id of object (use object_id from the object list),
      "arch_id": id of architecture element (use arch_id from the arch list),
      "adherence": true or false (indicating if object is placed against the architecture element),
    }}
  ]
}}
"""


SG_INIT_AUG_PROMPT = """Please follow the examples in the Visual Genome dataset and generate a scene graph that best describes the image. A reference list of labelled objects is provided as "{objects_with_ids}" in the format of <object_id, object_name>.
Return the output in JSON according to the following format:
{{
  "room_type": type of room,
  "objects": [
    {{
      "id": id of object (use object_id from the reference list),
      "name": name of object,
      "attributes": array of string,
      "caption": brief description of object appearance (excluding relationship),
    }}
  ],
  "support_relation": [
    {{
        "subject_id": id of object (use each object_id from the reference list)
        "type": choose from a list of supporting relationships "{support_rel}"
        "target_id": id of supporting object,
    }}
  ]
  "spatial_relation": [
    {{
        "subject_id": id of object,
        "type": choose from a list of spatial relationships "{spatial_rel}"
        "target_id": id of object,
    }}
  ]
}}
Relationships are expressed as <subject_id, type, target_id>. Each pair of <subject_id, target_id> only occurs once.
"""


SG_INIT_PROMPT = """Please follow the examples in the Visual Genome dataset and generate a scene graph that best describes the image.
Return the output in a JSON format according to the following format:
{{
  "room_type": type of room, such as bedroom,
  "objects": [
    {{
      "id": id of object,
      "name": name of object as string,
      "plural": true or false indicating if object is plural,
      "attributes": array of string,
      "caption": brief description of object appearance (excluding relationship),
    }}
  ],
  "relationships": [
    {{
        "type": type of relationship as string,
        "subject_id": id of object which is the subject of the relationship,
        "target_id": id of object which is the target of the relationship,
        "support": true or false indicating if subject object supports target object
    }}
  ]
}}

Tips:
1. Includes architectural objects such as floor and wall. Specify what objects are placed on floor.
2. The object and relationship IDs should start with 0 and increment. Every subject_id and target_id in relationships should correspond to an existing object ID.
3. If a number of objects are specified, please include each object in the count as a separate node (e.g, if the text specifies "two chairs", include two separate nodes for the chairs). And also describe the relationship among these objects.
"""


# OBJECT_PROMTP = """1. Can the object in the image recognized as {object}?
# 2. Can 6DoF pose of the object ({object}) in the image be estimated?
# Return the output in a JSON format according to the following format:
# {{
#   "shape_reason": A list of bullet points for object recognition reason,
#   "shape_recog": 0(no) or 1(yes) for object recognition result based on the reason,
#   "pose_reason": A list of bullet points for pose estimation reason,
#   "pose_recog": 0(no) or 1(yes) for pose estimation result based on the reason,
# }}
# """


OBJECT_PROMTP = """1. Can the object in the image recognized as {object}?
2. Can 6DoF pose of the object ({object}) in the image be estimated?
Return the output in a JSON format according to the following format:
{{
  "shape_recog": 0(no) or 1(yes) for object recognition,
  "pose_recog": 0(no) or 1(yes) for pose estimation,
}}
"""