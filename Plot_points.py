import numpy as np
import matplotlib.pyplot as plt
import csv


def read_landmarks_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    landmarks = []
    for line in lines:
        if line.strip().startswith('LANDMARKS1'):
            continue
        parts = line.split()
        if len(parts) == 6:  # Ensure the line has enough values
            landmarks.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(landmarks)

def read_landmarks_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
    
    # Convert the first row into a numpy array of x, y, z triplets
    first_row = np.array([float(val) for val in rows[0]]).reshape(-1, 3)
    return first_row

def plot_landmarks(txt_landmarks, femur_landmarks, tibia_landmarks):
    # Bones as defined in 4D_MSK py script
    femur = txt_landmarks[0:9, :]
    tibia = txt_landmarks[9:15, :]
    patella = txt_landmarks[15:22, :]
    extra_femur = txt_landmarks[22:25, :]
    
    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(femur[:, 0], femur[:, 1], femur[:, 2], c='red', label='Femur')
    ax.scatter(tibia[:, 0], tibia[:, 1], tibia[:, 2], c='blue', label='Tibia')
    ax.scatter(patella[:, 0], patella[:, 1], patella[:, 2], c='green', label='Patella')
    ax.scatter(extra_femur[:, 0], extra_femur[:, 1], extra_femur[:, 2], c='purple', label='Extra Femur')
    
    # Points from mypts.csv files
    ax.scatter(femur_landmarks[:, 0], femur_landmarks[:, 1], femur_landmarks[:, 2], 
               c='orange', label='femur First Row', marker='x')
    ax.scatter(tibia_landmarks[:, 0], tibia_landmarks[:, 1], tibia_landmarks[:, 2], 
               c='black', label='tibia First Row', marker='x')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title("3D Bone Landmarks and CSV Points")
    
    plt.show()

# File paths
txt_file_path = "Sample_DATA/HS_025/Left/points/points.txt"
femur_file_path = "Sample_DATA/HS_025/Left/points/Femur_mypts.csv"
tibia_file_path = "Sample_DATA/HS_025/Left/points/Tibia_mypts.csv"

# Read landmarks
txt_landmarks = read_landmarks_from_txt(txt_file_path)
femur_landmarks = read_landmarks_from_csv(femur_file_path)
tibia_landmarks = read_landmarks_from_csv(tibia_file_path)

# Plot landmarks
plot_landmarks(txt_landmarks, femur_landmarks, tibia_landmarks)
