import trimesh
import pyrender
import sys

def visualize_obj(file_path):
    print("Loading 3D model...")
    mesh = trimesh.load(file_path)

    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)

    # Convert trimesh to pyrender mesh
    render_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(render_mesh)

    viewer = pyrender.Viewer(scene, use_raymond_lighting=True)

# Call the function (set path to your .obj file)
if __name__ == "__main__":
    obj_path = "output/model_from_image.obj"
    visualize_obj(obj_path)
