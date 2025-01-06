import bpy
import os


def import_obj(file_path):
    # Import an OBJ file into the scene
    bpy.ops.wm.obj_import(filepath=file_path, import_vertex_groups = True)


def export_ply(obj, file_path):
    # Export an object as PLY
    bpy.ops.wm.ply_export(filepath=file_path, check_existing=True, export_attributes=True)


def process_folder(folder_path, output_folder):
    files = [f for f in os.listdir(folder_path) if f.endswith('.obj')]

    group_to_label = {
        "head": 1.0,
        "neck": 2.0,
        "torso": 3.0,
        "left_arm": 4.0,
        "right_arm": 5.0,
        "hip": 6.0,
        "legs": 7.0,
    }

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        import_obj(file_path)

        obj = bpy.context.selected_objects[0]  # Assume the imported object is the one to process

        if obj.type == 'MESH' and obj.vertex_groups:
            mesh = obj.data

            # Create a custom attribute 'label' if it doesn't exist
            if "label" not in mesh.attributes:
                label_layer = mesh.attributes.new(name="label", type='FLOAT', domain='POINT')

            # Assign label values based on vertex groups
            label_attr = mesh.attributes["label"].data
            for group_name, label_value in group_to_label.items():
                vertex_group = obj.vertex_groups.get(group_name)
                if vertex_group:
                    for vertex in mesh.vertices:
                        for group in vertex.groups:
                            if group.group == vertex_group.index:
                                label_attr[vertex.index].value = label_value
                                break
                else:
                    print(f"Vertex group '{group_name}' not found.")

            # Remove all vertex groups after assigning labels
            for vg in obj.vertex_groups:
                obj.vertex_groups.remove(vg)

        output_file_name = os.path.splitext(file_name)[0] + '.ply'
        output_file_path = os.path.join(output_folder, output_file_name)
        export_ply(obj, output_file_path)

        # Clear the scene after processing each file
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.ops.object.delete()

        print(f"Processed {file_name} and saved as {output_file_name}")


# Paths (update these to your actual paths)
folder_path = '../labeled/objs/raw'
output_folder = '../labeled/plys'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process all OBJ files in the folder
process_folder(folder_path, output_folder)

print("All files have been processed.")