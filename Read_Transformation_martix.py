import os
import numpy as np

# Function to load transformation matrices from a specified folder
def load_transformation_matrices(folder_path):
    matrices = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith("_tx.txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                matrix = np.loadtxt(file)
                matrices.append(matrix)
    return matrices

path = "DATA/HS_025/Left/Output/"
# path = "DATA/HS_027/Left/Output/"
# path = "DATA/HS_028/Left/Output/"

# HS_027
tibia_path = path + "Tibia_final_r3/Transform/"
femur_path = path + "Femur_final_r3/Transform/"
patella_path = path + "Patella_final_r3/Transform/"

# Load transformation matrices for each body part
Tibia_transformation = load_transformation_matrices(tibia_path)
Femur_transformation = load_transformation_matrices(femur_path)
Patella_transformation = load_transformation_matrices(patella_path)

# Display the results (optional)
print("Tibia Transformation Matrices:")
for i, matrix in enumerate(Tibia_transformation):
    print(f"Matrix {i}:\n{matrix}\n")

print("Femur Transformation Matrices:")
for i, matrix in enumerate(Femur_transformation):
    print(f"Matrix {i}:\n{matrix}\n")

print("Patella Transformation Matrices:")
for i, matrix in enumerate(Patella_transformation):
    print(f"Matrix {i}:\n{matrix}\n")