import torch
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
import open3d as o3d
import numpy as np
from PIL import Image

def process_text(prompt):
    print(f"Generating 3D point cloud for: '{prompt}'")

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_name = 'base40M-textvec'
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.load_state_dict(load_checkpoint(base_name, device))
    base_model.eval()

    sampler = PointCloudSampler(
        device=  device,
        models=  [base_model],
        diffusion=  diffusion_from_config(DIFFUSION_CONFIGS[base_name]),
        num_points=1024,
        aux_channels=['R', 'G', 'B'],
        guidance_scale=3.0,
    )

    samples = None
    for x in sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt])):
        samples = x

    pc = sampler.output_to_point_clouds(samples)[0]


    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(pc.coords.cpu().numpy())
    o3d_pc.colors = o3d.utility.Vector3dVector(pc.colors.cpu().numpy())
    
    o3d.io.write_point_cloud("output/text_to_3d.ply", o3d_pc)
    print("[DONE] Saved to output/text_to_3d.ply")


    o3d.visualization.draw_geometries ([o3d_pc])




