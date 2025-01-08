import numpy as np
import sys

args = sys.argv

mesh_file_path = args[1]
labels_file_path = args[2]

# Read mesh file as file
mesh_file = open(mesh_file_path, "r")
labels_file = open(labels_file_path, "r")

colors = []

# Read mesh
vertex = []
faces = []

while True:
    line = mesh_file.readline()
    if not line:
        break
    if line.startswith("v "):
        # Parse the position and color of the vertex
        x = float(line.split()[1])
        y = float(line.split()[2])
        z = float(line.split()[3])

        r = float(line.split()[4])
        g = float(line.split()[5])
        b = float(line.split()[6])

        vertex.append((x, y, z, r, g, b))
    if line.startswith("f "):
        # Parce the line to get the vertices
        v1 = int(line.split()[1])
        v2 = int(line.split()[2])
        v3 = int(line.split()[3])
        faces.append((v1, v2, v3))

# Close the File

mesh_file.close()
labels = []

# Read labels
while True:
    line = labels_file.readline()
    if not line:
        break

    labels.append(int(line))


# Generate a color for each unique labels
unique_labels = np.unique(labels)

colors = []
for i in range(len(unique_labels)):
    colors.append((np.random.rand(), np.random.rand(), np.random.rand()))

# For each face, get the label and color and assign the color to the vertices
for i in range(len(faces)):
    vertices = faces[i]
    label = labels[i]
    color = colors[label - 1]

    for v in vertices:
        x, y, z, r, g, b = vertex[v - 1]
        vertex[v - 1] = (x, y, z, color[0], color[1], color[2])

# Save the obj file
new_mesh_file = open("new_mesh.obj", "w")
for v in vertex:
    new_mesh_file.write(f"v {v[0]} {v[1]} {v[2]} {v[3]} {v[4]} {v[5]}\n")

for face in faces:
    new_mesh_file.write(f"f {face[0]} {face[1]} {face[2]}\n")

new_mesh_file.close()
