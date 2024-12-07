import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from Read_Transformation_martix import Tibia_transformation, Femur_transformation, Patella_transformation

# Function to extract translation vector from a 4x4 transformation matrix
# Function to extract translation vectors from a list of 4x4 transformation matrices
def extract_translation_vectors(transformation_matrices):
    translation_vectors = []
    for matrix in transformation_matrices:
        translation_vectors.append(matrix[:3, -1])  # Extract translation part
    return np.array(translation_vectors)

# Function to extract rotation matrix from a 4x4 transformation matrix
def extract_rotation_matrix(transformation_matrix):
    return transformation_matrix[:3, :3]

# Function to process transformations into Cardan (Euler) angles
def compute_cardan_angles(transformation_matrices, sequence="xyz"):
    cardan_angles = []
    for matrix in transformation_matrices:
        rotation_matrix = extract_rotation_matrix(matrix)
        r = R.from_matrix(rotation_matrix)  # Create a Rotation object
        angles = r.as_euler(sequence, degrees=True)  # Convert to Euler angles (degrees)
        cardan_angles.append(angles)
    return np.array(cardan_angles)

#  Function to calculate the relative homogeneous matrix between two transformation matrices
def calculate_relative_homogeneous_matrix(T1, T2):
    T1_inv = np.linalg.inv(T1)
    T_rel = []
    for i in range(len(T1_inv)):
        T_rel.append(np.dot(T1_inv[i], T2[i]))
    return np.array(T_rel)

T_rel = calculate_relative_homogeneous_matrix(Femur_transformation, Tibia_transformation)

relative_angles = compute_cardan_angles(T_rel)

# Compute Cardan angles for Tibia, Femur, and Patella
tibia_angles = compute_cardan_angles(Tibia_transformation)
femur_angles = compute_cardan_angles(Femur_transformation)
patella_angles = compute_cardan_angles(Patella_transformation)

tibia_translation = extract_translation_vectors(Tibia_transformation)
print(tibia_translation)
femur_translation = extract_translation_vectors(Femur_transformation)


# Plot Cardan angles
time_points = np.arange(len(tibia_angles))

plt.figure(figsize=(12, 8))

# Tibia
plt.subplot(3, 1, 1)
plt.plot(time_points, tibia_angles[:, 0], label='Roll (X)', color='r')
plt.plot(time_points, tibia_angles[:, 1], label='Pitch (Y)', color='g')
plt.plot(time_points, tibia_angles[:, 2], label='Yaw (Z)', color='b')
plt.title("Tibia Cardan Angles")
plt.xlabel("Time Points")
plt.ylabel("Angle (degrees)")
plt.legend()

# Femur
plt.subplot(3, 1, 2)
plt.plot(time_points, femur_angles[:, 0], label='Roll (X)', color='r')
plt.plot(time_points, femur_angles[:, 1], label='Pitch (Y)', color='g')
plt.plot(time_points, femur_angles[:, 2], label='Yaw (Z)', color='b')
plt.title("Femur Cardan Angles")
plt.xlabel("Time Points")
plt.ylabel("Angle (degrees)")
plt.legend()

# Patella
plt.subplot(3, 1, 3)
plt.plot(time_points, patella_angles[:, 0], label='Roll (X)', color='r')
plt.plot(time_points, patella_angles[:, 1], label='Pitch (Y)', color='g')
plt.plot(time_points, patella_angles[:, 2], label='Yaw (Z)', color='b')
plt.title("Patella Cardan Angles")
plt.xlabel("Time Points")
plt.ylabel("Angle (degrees)")
plt.legend()

#  Relative Angles
plt.figure(figsize=(12, 4))
plt.plot(time_points, relative_angles[:, 0], label='Roll (X)', color='r')
plt.plot(time_points, relative_angles[:, 1], label='Pitch (Y)', color='g')
plt.plot(time_points, relative_angles[:, 2], label='Yaw (Z)', color='b')
plt.title("Relative Cardan Angles (Femur - Tibia)")
plt.xlabel("Time Points")
plt.ylabel("Angle (degrees)")
plt.legend()

#  scatter tibia and femur translation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tibia_translation[:, 0], tibia_translation[:, 1], tibia_translation[:, 2], color='r', label='Tibia')
ax.scatter(femur_translation[:, 0], femur_translation[:, 1], femur_translation[:, 2], color='b', label='Femur')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()


# Display the plots
plt.tight_layout()
plt.show()
