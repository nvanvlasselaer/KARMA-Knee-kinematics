import numpy as np
import matplotlib.pyplot as plt
import csv

from Choose_subject import *

# Function to read landmarks from the text file
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

# Function to read and subdivide femur landmarks from the CSV file
def read_and_subdivide_femur_landmarks(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
    
    # Convert the first row into a numpy array of x, y, z triplets
    femur_landmarks = np.array([float(val) for val in rows[0]]).reshape(-1, 3)

    # Subdivide into specified categories
    subdivisions = {
        "Femur_EpiM": femur_landmarks[0],
        "Femur_EpiL": femur_landmarks[1],
        "Femur_Center_diaf": femur_landmarks[2],
        "Femur_Post_EpiM": femur_landmarks[3],
        "Femur_Post_EpiL": femur_landmarks[4],
        "Femur_TGroove": femur_landmarks[5],
        "Femur_surface_point_patella_GCS": femur_landmarks[6],
        "Femur_med_cond_surface_point_GCS": femur_landmarks[7],
        "Femur_lat_cond_surface_point_GCS": femur_landmarks[8],
        "Femur_medial_TG": femur_landmarks[9],
        "Femur_lateral_TG": femur_landmarks[10],
    }
    return subdivisions

# Function to read and subdivide tibia landmarks from the CSV file
def read_and_subdivide_tibia_landmarks(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
    
    # Convert the first row into a numpy array of x, y, z triplets
    tibia_landmarks = np.array([float(val) for val in rows[0]]).reshape(-1, 3)

    # Subdivide into specified categories
    subdivisions = {
        "Tibia_ConM": tibia_landmarks[0],
        "Tibia_ConL": tibia_landmarks[1],
        "Tibia_Center_diaf": tibia_landmarks[2],
        "Tibia_med_cond_surface_point_GCS": tibia_landmarks[3],
        "Tibia_lat_cond_surface_point_GCS": tibia_landmarks[4],
        "Tibia_TT": tibia_landmarks[5],
    }
    return subdivisions

# Function to read and subdivide patella landmarks from the CSV file
def read_and_subdivide_patella_landmarks(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
    
    # Convert the first row into a numpy array of x, y, z triplets
    patella_landmarks = np.array([float(val) for val in rows[0]]).reshape(-1, 3)

    # Subdivide into specified categories
    subdivisions = {
        "Patella_Ant": patella_landmarks[0],
        "Patella_Post": patella_landmarks[1],
        "Patella_Sup": patella_landmarks[2],
        "Patella_Inf": patella_landmarks[3],
        "Patella_Med": patella_landmarks[4],
        "Patella_Lat": patella_landmarks[5],
        "Patella_surface_point_GCS": patella_landmarks[6],
    }
    return subdivisions

# Plotting function
def plot_landmarks(txt_landmarks, femur_landmarks, tibia_landmarks):
    # Define the bone ranges for the text file landmarks
    femur = txt_landmarks[0:9, :]
    tibia_txt = txt_landmarks[9:15, :]
    patella = txt_landmarks[15:22, :]
    extra_femur = txt_landmarks[22:25, :]
    
    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the landmarks from the text file
    ax.scatter(femur[:, 0], femur[:, 1], femur[:, 2], c='red', label='Femur')
    ax.scatter(tibia_txt[:, 0], tibia_txt[:, 1], tibia_txt[:, 2], c='blue', label='Tibia')
    ax.scatter(patella[:, 0], patella[:, 1], patella[:, 2], c='green', label='Patella')
    ax.scatter(extra_femur[:, 0], extra_femur[:, 1], extra_femur[:, 2], c='purple', label='Extra Femur')
    
    # Add lines for femur
    ax.plot([femur_landmarks["Femur_EpiM"][0], femur_landmarks["Femur_EpiL"][0]],
            [femur_landmarks["Femur_EpiM"][1], femur_landmarks["Femur_EpiL"][1]],
            [femur_landmarks["Femur_EpiM"][2], femur_landmarks["Femur_EpiL"][2]], c='black', label="Femur EpiM-EpiL Line")
    
    ax.plot([femur_landmarks["Femur_Center_diaf"][0], femur_landmarks["Femur_TGroove"][0]],
            [femur_landmarks["Femur_Center_diaf"][1], femur_landmarks["Femur_TGroove"][1]],
            [femur_landmarks["Femur_Center_diaf"][2], femur_landmarks["Femur_TGroove"][2]], c='brown', label="Femur Diaf-TGroove Line")
    
    # Add lines for tibia
    ax.plot([tibia_landmarks["Tibia_ConM"][0], tibia_landmarks["Tibia_ConL"][0]],
            [tibia_landmarks["Tibia_ConM"][1], tibia_landmarks["Tibia_ConL"][1]],
            [tibia_landmarks["Tibia_ConM"][2], tibia_landmarks["Tibia_ConL"][2]], c='magenta', label="Tibia ConM-ConL Line")
    
    ax.plot([tibia_landmarks["Tibia_Center_diaf"][0], tibia_landmarks["Tibia_TT"][0]],
            [tibia_landmarks["Tibia_Center_diaf"][1], tibia_landmarks["Tibia_TT"][1]],
            [tibia_landmarks["Tibia_Center_diaf"][2], tibia_landmarks["Tibia_TT"][2]], c='blue', label="Tibia Diaf-TT Line")
    
    ax.plot([patella_landmarks["Patella_Med"][0], patella_landmarks["Patella_Lat"][0]],
            [patella_landmarks["Patella_Med"][1], patella_landmarks["Patella_Lat"][1]],
            [patella_landmarks["Patella_Med"][2], patella_landmarks["Patella_Lat"][2]], c='orange', label="Patella Med-Lat Line")
    
    ax.plot([patella_landmarks["Patella_Sup"][0], patella_landmarks["Patella_Inf"][0]],
            [patella_landmarks["Patella_Sup"][1], patella_landmarks["Patella_Inf"][1]],
            [patella_landmarks["Patella_Sup"][2], patella_landmarks["Patella_Inf"][2]], c='orange', label="Patella Sup-Inf Line")

    # Add labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title("3D Bone Landmarks with Connecting Lines")
    
    plt.show()

# File paths
txt_file_path = path + "points/points.txt"
femur_csv_path = path + "points/Femur_mypts.csv"
tibia_csv_path = path + "points/Tibia_mypts.csv"
patella_csv_path = path + "points/Patella_mypts.csv"

# Read landmarks from files
txt_landmarks = read_landmarks_from_txt(txt_file_path)
femur_landmarks = read_and_subdivide_femur_landmarks(femur_csv_path)
tibia_landmarks = read_and_subdivide_tibia_landmarks(tibia_csv_path)
patella_landmarks = read_and_subdivide_patella_landmarks(patella_csv_path)

# Plot landmarks
plot_landmarks(txt_landmarks, femur_landmarks, tibia_landmarks)
