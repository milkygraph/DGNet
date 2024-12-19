"""for script classes obj export by hand"""
import bpy

def export_mesh(obj, obj_path):
    # Export to OBJ (including vertex groups)
    bpy.ops.wm.obj_export(
        filepath=obj_path,
        check_existing=True,
        export_vertex_groups=True
    )

# Get the active object
active_obj = bpy.context.object

# Set the file paths for OBJ and PLY
obj_file_path = "../labeled/script_obj_with_classes.obj"

# Call the export function
export_mesh(active_obj, obj_file_path)

print("Export completed.")
