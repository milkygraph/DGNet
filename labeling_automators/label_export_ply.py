"""for script exporting ply labeled """

import bpy

def export_mesh(obj, ply_path):
    # Export to PLY (including vertex attributes)
    bpy.ops.wm.ply_export(
        filepath=ply_path,
        check_existing=True,
        export_attributes=True,
        # remove UV maps
        export_uv=False
    )

# Define encoding for vertex groups
group_to_label = {
    "head": 1.0,
    "neck": 2.0,
    "torso": 3.0,
    "left_arm": 4.0,
    "right_arm": 5.0,
    "hip": 6.0,
    "legs": 7.0,
}

# Get the active object
obj = bpy.context.object

# Ensure the object is a mesh and has vertex groups
if obj.type == 'MESH' and obj.vertex_groups:
    # Get the mesh data
    mesh = obj.data

    # Create a custom attribute 'label' if it doesn't exist
    if "label" not in mesh.attributes:
        mesh.attributes.new(name="label", type='FLOAT', domain='POINT')

    # Access the label attribute
    label_attr = mesh.attributes["label"].data

    # Iterate over the defined vertex groups and their labels
    for group_name, label_value in group_to_label.items():
        # Get the vertex group
        vertex_group = obj.vertex_groups.get(group_name)
        if vertex_group:
            # Iterate through vertices and assign the label
            for vertex in mesh.vertices:
                # Check if the vertex is in the vertex group
                for group in vertex.groups:
                    if group.group == vertex_group.index:
                        # Assign the label value
                        label_attr[vertex.index].value = label_value
                        break
        else:
            print(f"Vertex group '{group_name}' not found.")

    # Remove all vertex groups
    for vg in obj.vertex_groups:
        obj.vertex_groups.remove(vg)

else:
    print("Active object is not a mesh or has no vertex groups.")

# Set the file paths for OBJ and PLY
ply_file_path = "../labeled/script_ply_with_label.ply"

# Call the export function
export_mesh(obj, ply_file_path)

print("Export completed.")
