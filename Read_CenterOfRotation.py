import os
import re
from Choose_subject import *

# Regex to match the Origin pattern
origin_pattern = re.compile(r"\(CenterOfRotationPoint ([\d\.\-e]+) ([\d\.\-e]+) ([\d\.\-e]+)\)")

# Function to load Origin values from TransformParameters.0.txt files in a specified folder
def load_origin_values(folder_path):
    origins = []
    # Walk through the folder recursively
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name == "TransformParameters.0.txt":
                file_path = os.path.join(root, file_name)
                with open(file_path, 'r') as file:
                    content = file.read()
                    match = origin_pattern.search(content)
                    if match:
                        # Extract and convert the Origin values to a tuple of floats
                        origin_values = tuple(map(float, match.groups()))
                        origins.append(origin_values)
                        print(f"Processing file: {file_path}")

    return origins


tibia_path = os.path.join(path, "Output/Tibia_final_r3/")
femur_path = os.path.join(path, "Output/Femur_final_r3/")
patella_path = os.path.join(path, "Output/Patella_final_r3/")

# Load Origin values for each body part
Tibia_origins = load_origin_values(tibia_path)
Femur_origins = load_origin_values(femur_path)
Patella_origins = load_origin_values(patella_path)

print("Tibia Origins:")
for i, origin in enumerate(Tibia_origins):
    print(f"Origin {i}: {origin}")

print("\nFemur Origins:")
for i, origin in enumerate(Femur_origins):
    print(f"Origin {i}: {origin}")

print("\nPatella Origins:")
for i, origin in enumerate(Patella_origins):
    print(f"Origin {i}: {origin}")

#  plot the origins as scatter points in 3D space
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Extract x, y, z coordinates from the origins
tibia_x, tibia_y, tibia_z = zip(*Tibia_origins)
femur_x, femur_y, femur_z = zip(*Femur_origins)
patella_x, patella_y, patella_z = zip(*Patella_origins)

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.scatter(tibia_x, tibia_y, tibia_z, color='r', label='Tibia')
ax.scatter(femur_x, femur_y, femur_z, color='g', label='Femur')
ax.scatter(patella_x, patella_y, patella_z, color='b', label='Patella')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()
