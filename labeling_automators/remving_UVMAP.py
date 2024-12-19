# note you can just export without the UV maps this script is useless for now

import bpy

# Function to remove a UV Map from the active object
def remove_uv_map(obj, uv_map_name):
    # Access the mesh data
    mesh = obj.data

    # Check if the UV Map exists
    if uv_map_name in mesh.uv_layers:
        # Remove the UV Map
        mesh.uv_layers.remove(mesh.uv_layers[uv_map_name])


# Get the active object
obj = bpy.context.object

# Specify the name of the UV Map to remove
uv_map_name = "UVMap"  # Change this to the exact name of your UV Map

# Call the function to remove the UV Map
remove_uv_map(obj, uv_map_name)

print("UV Map removed.")
