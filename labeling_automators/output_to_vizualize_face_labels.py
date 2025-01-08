import bpy
import bmesh


def create_vertex_groups_from_labels(obj_file, label_file):
    """
    Imports an OBJ file and creates vertex groups in Blender based on face labels.
    Each face's vertices will be assigned to a vertex group corresponding to its label.

    Parameters:
        obj_file (str): Path to the OBJ file.
        label_file (str): Path to the text file containing face labels.
    """
    # Import the OBJ file into Blender
    bpy.ops.wm.obj_import(filepath=obj_file)
    obj = bpy.context.selected_objects[0]

    # Read face labels from the text file
    with open(label_file, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    # Switch to Object Mode
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='OBJECT')

    # Create BMesh for better face handling
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    # Create vertex groups for each unique label
    unique_labels = sorted(set(labels))
    vertex_groups = {label: obj.vertex_groups.new(name=f"Label_{label}") for label in unique_labels}

    # Assign vertices of each face to corresponding vertex groups
    for face_idx, face in enumerate(bm.faces):
        if face_idx >= len(labels):
            print(f"Warning: More faces in mesh than labels in file")
            break

        label = labels[face_idx]
        vertex_group = vertex_groups[label]

        # Get indices of vertices in this face
        vert_indices = [v.index for v in face.verts]

        # Assign vertices to the vertex group
        vertex_group.add(vert_indices, 1.0, 'ADD')

    # Clean up BMesh
    bm.free()

    # Create material slots and materials for visualization
    for label in unique_labels:
        # Create new material
        mat_name = f"Material_Label_{label}"
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True

        # Set random color for the material
        nodes = mat.node_tree.nodes
        principled = nodes.get('Principled BSDF')
        if principled:
            # Create somewhat random but visually distinct colors
            hue = (hash(label) % 100) / 100.0
            principled.inputs['Base Color'].default_value = (hue, 0.7, 0.7, 1.0)

        # Assign material to object
        obj.data.materials.append(mat)

    print(f"Vertex groups and materials created for object: {obj.name}")


def visualize_face_labels(obj):
    """
    Helper function to visualize the face labels by selecting faces
    in different vertex groups.

    Parameters:
        obj: Blender object with vertex groups
    """
    # Switch to edit mode
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    # Deselect all
    bpy.ops.mesh.select_all(action='DESELECT')

    # Store current selection mode
    select_mode = bpy.context.tool_settings.mesh_select_mode[:]

    # Set face select mode
    bpy.context.tool_settings.mesh_select_mode = (False, False, True)

    # For each vertex group, select faces and assign material
    for i, vertex_group in enumerate(obj.vertex_groups):
        # Select vertices in group
        bpy.ops.object.vertex_group_set_active(group=vertex_group.name)
        bpy.ops.object.vertex_group_select()

        # Assign material to selected faces
        if i < len(obj.material_slots):
            obj.active_material_index = i
            bpy.ops.object.material_slot_assign()

    # Restore original selection mode
    bpy.context.tool_settings.mesh_select_mode = select_mode

    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')


# Example usage
if __name__ == "__main__":
    # Replace these paths with your actual file paths
    obj_file = "raw_60_2.obj"
    label_file = "raw_60_2_face_labels.txt"

    # Create vertex groups and materials
    create_vertex_groups_from_labels(obj_file, label_file)

    # Visualize the labels
    obj = bpy.context.active_object
    visualize_face_labels(obj)
