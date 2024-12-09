import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv

from Choose_subject import *

# Functions to read and subdivide landmarks from the CSV files: femur, tibia, and patella
def read_and_subdivide_femur_landmarks(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
    landmarks = [np.array([float(val) for val in row]).reshape(-1, 3) for row in rows]
    return landmarks

def read_and_subdivide_tibia_landmarks(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
    landmarks = [np.array([float(val) for val in row]).reshape(-1, 3) for row in rows]
    return landmarks

def read_and_subdivide_patella_landmarks(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
    landmarks = [np.array([float(val) for val in row]).reshape(-1, 3) for row in rows]
    return landmarks

# Plot initialization
def init_plot():
    ax.clear()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("3D Bone Landmarks Animation")
    ax.set_xlim(-50, 250)  # Adjust these limits as per your data
    ax.set_ylim(-150, 150)
    ax.set_zlim(-150, 150)

# Update function for animation
def update(frame):
    init_plot()

    # Subdivide landmarks for the current frame
    femur_sub = {
        "Femur_EpiM": femur_landmarks[frame][0],
        "Femur_EpiL": femur_landmarks[frame][1],
        "Femur_Center_diaf": femur_landmarks[frame][2],
        "Femur_Post_EpiM": femur_landmarks[frame][3],
        "Femur_Post_EpiL": femur_landmarks[frame][4],
        "Femur_TGroove": femur_landmarks[frame][5],
        "Femur_surface_point_patella_GCS": femur_landmarks[frame][6],
        "Femur_med_cond_surface_point_GCS": femur_landmarks[frame][7],
        "Femur_lat_cond_surface_point_GCS": femur_landmarks[frame][8],
        "Femur_medial_TG": femur_landmarks[frame][9],
        "Femur_lateral_TG": femur_landmarks[frame][10],
    }

    tibia_sub = {
        "Tibia_ConM": tibia_landmarks[frame][0],
        "Tibia_ConL": tibia_landmarks[frame][1],
        "Tibia_Center_diaf": tibia_landmarks[frame][2],
        "Tibia_med_cond_surface_point_GCS": tibia_landmarks[frame][3],
        "Tibia_lat_cond_surface_point_GCS": tibia_landmarks[frame][4],
        "Tibia_TT": tibia_landmarks[frame][5],
    }

    patella_sub = {
        "Patella_Ant": patella_landmarks[frame][0],
        "Patella_Post": patella_landmarks[frame][1],
        "Patella_Sup": patella_landmarks[frame][2],
        "Patella_Inf": patella_landmarks[frame][3],
        "Patella_Med": patella_landmarks[frame][4],
        "Patella_Lat": patella_landmarks[frame][5],
        "Patella_surface_point_GCS": patella_landmarks[frame][6],
    }

    # Scatter and connect points for femur
    for landmark in femur_sub.values():
        ax.scatter(*landmark, c='red', marker='o')
    ax.plot(
        [femur_sub["Femur_EpiM"][0], femur_sub["Femur_EpiL"][0]],
        [femur_sub["Femur_EpiM"][1], femur_sub["Femur_EpiL"][1]],
        [femur_sub["Femur_EpiM"][2], femur_sub["Femur_EpiL"][2]],
        c='black', label="Femur EpiM-EpiL Line"
    )

    ax.plot(
        [femur_sub["Femur_Center_diaf"][0], femur_sub["Femur_TGroove"][0]],
        [femur_sub["Femur_Center_diaf"][1], femur_sub["Femur_TGroove"][1]],
        [femur_sub["Femur_Center_diaf"][2], femur_sub["Femur_TGroove"][2]],
        c='brown', label="Femur Diaf-TGroove Line"
    )

    # Scatter and connect points for tibia
    for landmark in tibia_sub.values():
        ax.scatter(*landmark, c='blue', marker='o')
    ax.plot(
        [tibia_sub["Tibia_ConM"][0], tibia_sub["Tibia_ConL"][0]],
        [tibia_sub["Tibia_ConM"][1], tibia_sub["Tibia_ConL"][1]],
        [tibia_sub["Tibia_ConM"][2], tibia_sub["Tibia_ConL"][2]],
        c='magenta', label="Tibia ConM-ConL Line"
    )

    ax.plot(
        [tibia_sub["Tibia_Center_diaf"][0], tibia_sub["Tibia_TT"][0]],
        [tibia_sub["Tibia_Center_diaf"][1], tibia_sub["Tibia_TT"][1]],
        [tibia_sub["Tibia_Center_diaf"][2], tibia_sub["Tibia_TT"][2]],
        c='cyan', label="Tibia Diaf-TT Line"
    )

    # Scatter and connect points for patella
    for landmark in patella_sub.values():
        ax.scatter(*landmark, c='green', marker='o')
    ax.plot(
        [patella_sub["Patella_Med"][0], patella_sub["Patella_Lat"][0]],
        [patella_sub["Patella_Med"][1], patella_sub["Patella_Lat"][1]],
        [patella_sub["Patella_Med"][2], patella_sub["Patella_Lat"][2]],
        c='orange', label="Patella Med-Lat Line"
    )


femur_csv_path = path + "points/Femur_mypts.csv"
tibia_csv_path = path + "points/Tibia_mypts.csv"
patella_csv_path = path + "points/Patella_mypts.csv"

# Read landmarks
femur_landmarks = read_and_subdivide_femur_landmarks(femur_csv_path)
tibia_landmarks = read_and_subdivide_tibia_landmarks(tibia_csv_path)
patella_landmarks = read_and_subdivide_patella_landmarks(patella_csv_path)

# Setup figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize animation
frames = min(len(femur_landmarks), len(tibia_landmarks), len(patella_landmarks))

# Create a boomerang sequence of frames
forward_frames = list(range(frames))
backward_frames = list(range(frames - 2, 0, -1))  # Exclude the last frame to avoid duplication
boomerang_frames = forward_frames + backward_frames

# Initialize animation with boomerang frames
ani = FuncAnimation(fig, update, frames=boomerang_frames, interval=500)

plt.show()
