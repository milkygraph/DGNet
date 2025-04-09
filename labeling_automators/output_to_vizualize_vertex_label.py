import bpy

def create_vertex_groups_from_vertex_labels(obj_file, label_file):
    """
    Imports an OBJ file into Blender and creates vertex groups based on vertex labels.
    Each label is used to group corresponding vertices into a vertex group.

    Parameters:
        obj_file (str): Path to the OBJ file.
        label_file (str): Path to the text file containing vertex labels.
    """
    # Import the OBJ file into Blender
    bpy.ops.wm.obj_import(filepath=obj_file, import_vertex_groups = True)
    obj = bpy.context.selected_objects[0]  # Assume the imported object is active

    # Read vertex labels from the text file
    with open(label_file, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    # Ensure the object is in Object Mode
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="OBJECT")

    # Check if the number of labels matches the number of vertices
    mesh = obj.data
    if len(labels) != len(mesh.vertices):
        raise ValueError(
            f"Mismatch: {len(labels)} labels provided, but the mesh has {len(mesh.vertices)} vertices."
        )

    # Create vertex groups for each unique label
    unique_labels = sorted(set(labels))
    vertex_groups = {label: obj.vertex_groups.new(name=f"Label_{label}") for label in unique_labels}

    # Assign each vertex to the appropriate vertex group based on its label
    for vert_idx, label in enumerate(labels):
        group = vertex_groups[label]
        group.add([vert_idx], 1.0, "ADD")

    # Create materials for visualization
    for label in unique_labels:
        # Create new material
        mat_name = f"Material_Label_{label}"
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True

        # Assign a random color for the material
        nodes = mat.node_tree.nodes
        principled = nodes.get("Principled BSDF")
        if principled:
            hue = (hash(label) % 100) / 100.0
            principled.inputs["Base Color"].default_value = (hue, 0.7, 0.7, 1.0)

        # Append material to the object
        obj.data.materials.append(mat)

    # Assign materials to vertices for visualization
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    for label, group in vertex_groups.items():
        # Select the vertices in the group
        bpy.ops.object.vertex_group_set_active(group=group.name)
        bpy.ops.object.vertex_group_select()

        # Assign the corresponding material
        mat_index = list(unique_labels).index(label)
        obj.active_material_index = mat_index
        bpy.ops.object.material_slot_assign()

    # Return to Object Mode
    bpy.ops.object.mode_set(mode="OBJECT")

    print(f"Vertex groups and materials created for object: {obj.name}")


# Example usage
if __name__ == "__main__":
    # Replace these paths with your actual file paths
    obj_file = "raw_60_2.obj"  # Path to your OBJ file
    label_file = "raw_60_2_vertex_labels.txt"  # Path to the vertex label file

    # Create vertex groups and visualize them
    create_vertex_groups_from_vertex_labels(obj_file, label_file)
