import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from rembg import remove
import trimesh
import open3d as o3d

def process_image(image_path):
    print("🔹 Loading image and removing background...")
    img = Image.open(image_path).convert("RGB")
    img_no_bg = remove(img).convert("RGB")
    os.makedirs("output", exist_ok=True)
    img_no_bg.save("output/cleaned.png")

    print("🔹 Estimating depth with MiDaS_small...")

    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)

    transform = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5])
    ])
    input_tensor = transform(img_no_bg).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = midas(input_tensor)
        depth = prediction.squeeze().cpu().numpy()

    depth -=  depth.min()
    depth /=  depth.max()

    print("🔹 Creating 3D point cloud...")

    img_np = np.array(img_no_bg.resize((256, 256)))
    h, w = depth.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    X = xx.astype(np.float32)
    Y = yy.astype(np.float32)
    Z = (depth * 255).astype(np.float32)

    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    colors = img_np.reshape(-1, 3) / 255.0  # Normalize to [0, 1] for Open3D

    mask = Z.flatten() > 0
    points = points[mask]
    colors = colors[mask]

    print("🔹 Saving point cloud to OBJ...")
    cloud = trimesh.points.PointCloud(points, colors=(colors * 255).astype(np.uint8))
    obj_path = "output/model_from_image.obj"
    cloud.export(obj_path)
    print(f" 3D point cloud saved to: {obj_path}")

    print("🔹 Visualizing with Open3D..." )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name="3D Point Cloud")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to image file (e.g., car.jpg)")
    args = parser.parse_args()
    process_image(args.input)

