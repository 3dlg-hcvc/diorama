# Diorama

### Diorama: Unleashing Zero-shot Single-view 3D Scene Modeling


[Qirui Wu](https://qiruiw.github.io/), [Denys Iliash](), [Daniel Ritchie](https://dritchie.github.io/), [Manolis Savva](https://msavva.github.io/), [Angel X. Chang](http://angelxuanchang.github.io/)


ICCV 2025

[Website](https://3dlg-hcvc.github.io/diorama/) | [arXiv](https://arxiv.org/abs/2411.19492) 

**TL;DR**: Our work is driven by the question *"Is holistic 3D scene modeling from a single-view real-world image possible using foundation models?"* To answer it, we present **Diorama: a modular zero-shot open-world system that models synthetic holistic 3D scenes given an image and requires no end-to-end training**.

![teaser](docs/static/images/teaser.png)


## Setup

```shell
# module load LIB/CUDA/12.1 LIB/CUDNN/8.8.0-CUDA12.0 # if necessary
git clone --recurse-submodules git@github.com:3dlg-hcvc/diorama.git
# create and activate the conda environment
conda create -n diorama python=3.10
conda activate diorama

# conda install pytorch==2.4.1 torchvision==0.19.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# pip3 install torch torchvision
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
cd third_party/sam2 && pip install -e . && cd ../.. # install SAM2
cd third_party/GroundingDINO && pip install -e . && cd ../.. # install gdino
# conda install xformers -c xformers
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121
pip install "git+https://github.com/facebookresearch/pytorch3d.git" # or conda install pytorch3d -c pytorch3d -c conda-forge
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl
# pip install "git+https://github.com/facebookresearch/detectron2.git"

cd third_party/open_clip_mod &&  pip install . && cd ../.. # install openclip

# download pretrained weights
mkdir -p weights && cd weights
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
cd ..

# install Python libraries
pip install -e .
```

## Model Weights
We use open-sourced pretrained model weights for different pipeline components:
- [DINOv2](https://github.com/facebookresearch/dinov2) `dinov2_vitl14_pretrain.pth` for image patch features extraction.
- [OWLv2](https://huggingface.co/google/owlv2-large-patch14-ensemble) `google/owlv2-large-patch14-ensemble` for open-world object detection.
- [SAMv2](https://github.com/facebookresearch/sam2) `sam2.1_hiera_large.pt` for object segmentation.
- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) `ZhengPeng7/BiRefNet` for dichotomous image segmentation.
- [Metric3Dv2](https://github.com/YvanYin/Metric3D) `metric3d_vit_large` for metric depth estimation.
- [DuoduoCLIP](https://github.com/3dlg-hcvc/DuoduoCLIP) `Four_1to6F_bs1600_LT6.ckpt` for CAD shape retrieval.
- Additionally, we leverge a samll scale estimation model from [GigaPose](https://github.com/nv-nguyen/gigapose) that we provide [here](https://huggingface.co/datasets/3dlg-hcvc/diorama).


## Usage

To have access to GPT4
```shell
export OPENAI_API_KEY=<openai_api_key>
```

Test the system on a single image by runing different components.
```shell
# open-world object recognition
python run.py exp_name=<exp_name> img_path=<img_path> vlm.sun=True

# depth estimation
# python run.py exp_name=<exp_name> img_path=<img_path> load_depth_model=True
python scripts/estimate_depth.py --img_path <path/to/image> --intrinsics <supported dataset name or custom>

# open-world 2D perception & LLM-powered scene graph generation
python run.py exp_name=<exp_name> img_path=<img_path> load_perception_model=True

# 3D shape retrieval
python run.py exp_name=<exp_name> img_path=<img_path> load_retrieval_model=True

# object pose estimation
python run.py exp_name=<exp_name> img_path=<img_path> load_pose_model=True

# scene layout optimization
python run.py exp_name=<exp_name> img_path=<img_path> run_optimization=True
```

### Architecture reconstruction

First, we need to obtain the dichotomous segmentation
```shell
python scripts/compute_dichotomous_segmentation.py --data_path <path_to_experiment> --output_path <path_to_experiment>
```

In order to inpaint the images, you need to run ```third_party/Inpaint-Anything/remove_anything_masks_naive_predicted_seg.py``` for predicted segmentation or ```third_party/Inpaint-Anything/remove_anything_masks_naive.py``` for GT segmentation. Consider modifying and running the wrapper script for this matter
```shell
python third_party/Inpaint-Anything/wss_inference_naive.py
```

Next, obtain the depth predictions by running 
```shell
python scripts/compute_inpainted_depth_normal.py --exp_path <path_to_experiment> --encoder vit_giant2 --intrinsics <supported dataset name or custom> --normals
```

We are now ready to run plane segmentation
```shell
python third_party/Inpaint-Anything/compute_plane_segmentation.py --exp_path <path_to_experiment> --pcd_type <pcd_name_exported_by_depth_script> --postfix <experiment_label>
```

Finally, configure and run the script to obtain the architecture planes

```shell
python diorama/utils/arch_util.py
```

#### Architecture reconstruction evaluation

Note, further steps require installing ```pointops```:
```shell
# PointOps from PointCept libbrary - https://github.com/Pointcept/Pointcept
cd libs/pointops
python setup.py install
```

For evaluation, you need to render the architectural planes and sample points from them and GT. Configure and run
```shell
python third_party/Inpaint-Anything/postprocess_render_arch.py
python third_party/Inpaint-Anything/postprocess_sample_points.py
python third_party/Inpaint-Anything/create_gt_points.py
```

Finally, run evaluation
```shell
python eval_arch_rec.py --exp_path <path_to_experiment> --segmentation_type <name_of_segmentaiton> --gt_path <path_to_gt> --gt_seg_path <path_to_gt_segmentation> --split val

python eval_arch_geo.py --exp_path <path_to_experiment> --segmentation_type <name_of_segmentaiton> --gt_points_path <path_to_gt_sampled_points> --pcd_type <name_of_pred_sampled_pcds> --split val
```


## Bibtex
```
@article{wu2024diorama,
  title={Diorama: Unleashing Zero-shot Single-view 3D Scene Modeling},
  author={Wu, Qirui and Iliash, Denys and Ritchie, Daniel and Savva, Manolis and Chang, Angel X},
  journal={arXiv preprint arXiv:2411.19492},
  year={2024}
}
```
