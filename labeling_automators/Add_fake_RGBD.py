import bpy


def add_fake_rgbd_to_mesh(obj, rgb=(255, 255, 255), depth=255):
    mesh = obj.data

    # Ensure the mesh has vertex colors
    if not mesh.vertex_colors:
        mesh.vertex_colors.new(name="Col", do_init=False)

    color_layer = mesh.vertex_colors["Col"]

    # Assign the RGB and Depth (as Alpha) values to each vertex
    for poly in mesh.polygons:
        for idx in poly.loop_indices:
            loop_color = color_layer.data[idx]
            # Normalize RGB and Depth to [0, 1] range and set Depth as Alpha
            loop_color.color = (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, depth / 255)


# Export to PLY with vertex colors
def export_mesh_ply(obj, ply_path):
    # Export to PLY (including vertex attributes)
    bpy.ops.wm.ply_export(
        filepath=ply_path,
        check_existing=True,
        export_attributes=True
    )


# Set up your object and file path
obj = bpy.context.object  # Ensure this is the mesh object you want to export
file_path = "../test/script_label_fake_color.ply"  # Update this path

# Add fake RGBD to the mesh
add_fake_rgbd_to_mesh(obj)
# Export the mesh with RGBD data
export_mesh_ply(obj, file_path)

print("Mesh with fake RGBD exported.")
