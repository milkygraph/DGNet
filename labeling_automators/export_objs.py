import bpy

def export_mesh(obj, obj_path):
    # Export to OBJ (including vertex groups)
    bpy.ops.wm.obj_export(
        filepath=obj_path,
        check_existing=True,
        export_vertex_groups=True
    )

def read_number(file_path):
    try:
        with open(file_path, 'r') as file:
            number = int(file.read().strip())
            return number
    except FileNotFoundError:
        return 1  # Return 1 if the file does not exist

def write_number(file_path, number):
    with open(file_path, 'w') as file:
        file.write(str(number))

# File path where the number is stored
number_file_path = '../labeled/sample_number.txt'

# Get the active object
active_obj = bpy.context.object

# Read the current number from the file
current_number = read_number(number_file_path)

# Set the file path for OBJ based on the current number
obj_file_path = f"../labeled/objs/raw/raw_{current_number}.obj"

# Call the export function
export_mesh(active_obj, obj_file_path)

# Increment the number and write it back to the file
write_number(number_file_path, current_number + 1)

print("Export completed.")