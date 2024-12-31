import bpy
import os


def import_obj(file_path):
    # Import an OBJ file into the scene
    bpy.ops.wm.import_obj(filepath=file_path, import_vertex_groups = True)


def export_ply(obj, file_path):
    # Export an object as PLY
    bpy.ops.wm.ply_export(filepath=file_path, check_existing=True, export_vertex_groups=True)


def process_folder(folder_path, output_folder):
    # List all files in the directory
    files = [f for f in os.listdir(folder_path) if f.endswith('.obj')]

    # Process each file
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)

        # Import the OBJ file
        import_obj(file_path)

        # Assume the imported object is active
        obj = bpy.context.selected_objects[0]  # Adjust if needed

        # Define the output file path (change extension to .ply)
        output_file_name = os.path.splitext(file_name)[0] + '.ply'
        output_file_path = os.path.join(output_folder, output_file_name)

        # Export the object as PLY
        export_ply(obj, output_file_path)

        # Optionally clear the scene after processing each file
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.ops.object.delete()

        print(f"Processed {file_name} and saved as {output_file_name}")


# Paths (update these to your actual paths)
folder_path = '/path/to/your/obj_files'
output_folder = '/path/to/output/ply_files'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process all OBJ files in the folder
process_folder(folder_path, output_folder)

print("All files have been processed.")
