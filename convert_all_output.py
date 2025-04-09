import os
import numpy as np
import sys


def convert_output(mesh_file_path, labels_file_path, output_file_path):
    """Processes a mesh file by assigning colors based on labels and saves the result."""

    # Read mesh file
    with open(mesh_file_path, "r") as mesh_file:
        vertex = []
        faces = []

        for line in mesh_file:
            if line.startswith("v "):
                parts = line.split()
                x, y, z = map(float, parts[1:4])
                r, g, b = map(float, parts[4:7])
                vertex.append((x, y, z, r, g, b))
            elif line.startswith("f "):
                v1, v2, v3 = map(int, line.split()[1:4])
                faces.append((v1, v2, v3))

    # Read labels file
    with open(labels_file_path, "r") as labels_file:
        labels = [int(line.strip()) for line in labels_file]

    # Generate colors for unique labels
    unique_labels = np.unique(labels)
    label_colors = {label: (np.random.rand(), np.random.rand(), np.random.rand()) for label in unique_labels}

    # Assign colors based on face labels
    for i, face in enumerate(faces):
        label = labels[i]
        color = label_colors[label]

        for v in face:
            x, y, z, _, _, _ = vertex[v - 1]
            vertex[v - 1] = (x, y, z, *color)

    # Save the new colored mesh
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as new_mesh_file:
        for v in vertex:
            new_mesh_file.write(f"v {v[0]} {v[1]} {v[2]} {v[3]} {v[4]} {v[5]}\n")
        for face in faces:
            new_mesh_file.write(f"f {face[0]} {face[1]} {face[2]}\n")

def process_directories(mesh_dir, labels_dir, output_dir):
    """Loops through all matching files in directories and processes them."""

    if not os.path.exists(mesh_dir) or not os.path.exists(labels_dir):
        print("Error: One or both input directories do not exist.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all mesh files
    for file in os.listdir(mesh_dir):
        if file.startswith("raw_") and file.endswith("_1.obj"):
            sample_number = file[len("raw_"):-len("_1.obj")]  # Extract sample number

            mesh_file_path = os.path.join(mesh_dir, file)
            labels_file_name = f"raw_{sample_number}_1_outputs.txt"
            labels_file_path = os.path.join(labels_dir, labels_file_name)

            if os.path.exists(labels_file_path):
                output_file_name = f"downsampled_colored_{sample_number}.obj"
                output_file_path = os.path.join(output_dir, output_file_name)
                print(f"Processing {file} -> {output_file_name}")
                convert_output(mesh_file_path, labels_file_path, output_file_path)
            else:
                print(f"Warning: No matching labels file for {file}")



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <mesh_directory> <labels_directory>")
        sys.exit(1)

    mesh_directory = sys.argv[1]
    labels_directory = sys.argv[2]
    output_directory = os.path.join(os.getcwd(), "colored_outputs")

    process_directories(mesh_directory, labels_directory, output_directory)

# horriable samples:  2, 7, 8, 12, 13, 14, 15