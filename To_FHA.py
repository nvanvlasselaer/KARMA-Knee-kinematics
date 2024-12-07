
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
from scipy.spatial.transform import Rotation as R


from Read_Transformation_martix import Tibia_transformation, Femur_transformation, Patella_transformation

plt.close('all')
    
T1 = np.array(Femur_transformation)
T2 = np.array(Tibia_transformation)
    
#%% Calculate relative homogenous matrix

T1inv=[]
for i in range(len(T1)):
    T1inv.append(np.linalg.inv(T1[i]))

T1inv = np.array(T1inv)
    

Trel = []
for i in range(len(T1inv)):
    # Note: we pre-multiply Tinv, meaning an EXTRINSIC rotation 
    # (result of multiple combined rotations is always referred to the orignal reference system) 
    Trel.append(np.dot(T1inv[i,:,:], T2[i,:,:]))

Trel=np.array(Trel)


#%% Calculate FHA    
def decompose_homogeneous_matrix(H):
    Horig = np.array(H)
    R = Horig[0:3,0:3]
    v = Horig[0:3,3].transpose()
    
    return R, v

def calculate_FHA(T1, T2):
    #Takes two homogeneous matrices as an input and returns parameters for the FHA associated with the rototranslation between them
    
    #Parameters returned:
        #n: normalized vector representing the direction of the FHA
        #phi: angle around the FHA
        #t: displacement along the FHA
        #s: location of the FHA (application point for the n vector)
    
    H = np.dot(T2, np.linalg.inv(T1))   #In this case, we post-multiply (INTRINSIC rotation: every subsequent rotation is
                                        #referred to the reference system of previous pose, and not to the global one)

    ROT, v = decompose_homogeneous_matrix(H)
    
    sinPhi = 0.5*np.sqrt(pow(ROT[2,1]-ROT[1,2],2)+pow(ROT[0,2]-ROT[2,0],2)+pow(ROT[1,0]-ROT[0,1],2))
    
    #CAREFUL: this calculation for cosine only works when sinPhi > sqrt(2)/2
    cosPhi=0.5*(np.trace(ROT)-1)
    
    #Implementing this condition, can use cosPhi calculated as before to estimate phi
    if sinPhi <= (np.sqrt(2)/2):
        # phi = math.degrees(np.arcsin(sinPhi))     #deg
        phi = np.arcsin(sinPhi)                     #rad
    else:
        # phi = math.degrees(np.arccos(cosPhi))     #deg
        phi = np.arccos(cosPhi)                     #rad
        
    n = (1/(2*sinPhi))*np.array([ROT[2,1]-ROT[1,2], ROT[0,2]-ROT[2,0], ROT[1,0]-ROT[0,1]])
    t = np.dot(n, np.array(v.transpose()))
    
    #The vector s (location of the FHA) should be calculated re-estimating sine and cosine of phi
    #through traditional functions (once phi is obtained), not using the sinPhi and cosPhi estimated from
    #the rotation matrix, because that calculation only works for sinPhi > sqrt(2)/2
    s = np.cross(-0.5*n, np.cross(n, v)) + np.cross((np.sin(phi)/(2*(1-np.cos(phi))))*n, v)
    
    return phi, n, t, s
    
hax = []
ang = []
svec = []
d = []

#  create variable to store the v values of T1 and T2
v1 = []
v2 = []
for i in range(len(T1)):
    ROT1, v1_ = decompose_homogeneous_matrix(T1[i])
    ROT2, v2_ = decompose_homogeneous_matrix(T2[i])
    v1.append(v1_)
    v2.append(v2_)

# Traditional method
for i in range(len(Trel)-1):
    phi, n, t, s = calculate_FHA(Trel[i], Trel[i+1])
    hax.append(n)
    ang.append(phi)
    svec.append(s)
    d.append(t)
 
    
#%% Plotting 

#transform into sensor1 reference system for plotting
R1=[]
for i in range(len(T1)):
    ROT,v = decompose_homogeneous_matrix(T1[i])
    R1.append(ROT)
R1=np.array(R1)

transformed_hax = []
transformed_svec = []
for i in range(len(hax)):
    transformed_hax.append(np.dot(hax[i], R1[i]))
    transformed_svec.append(np.dot(T1[i], np.append(svec[i], 1).transpose()))


#plot
fig = plt.figure()
axis_scale = 20  
ax = fig.add_subplot(111, projection='3d') 


for i in range(len(hax)):
    p = transformed_svec[i][0:3] + d[i]*transformed_hax[i]
    
    start = p 
    end = p + transformed_hax[i]*axis_scale
    
    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'r-')
    ax.scatter(p[0], p[1], p[2], color='b', s=5)

    # scatter v1 and v2
    ax.scatter(v1[i][0], v1[i][1], v1[i][2], color='g', s=5)
    ax.scatter(v2[i][0], v2[i][1], v2[i][2], color='y', s=5)
    
    
# ax.scatter(translation_2_list[0][0], translation_2_list[0][1], translation_2_list[0][2], color='k', s=50) 

#this is to align the plotting reference frame with the Polhemus transmitter reference frame (needed if data were acquired using the default reference system)
# ax.view_init(elev=180) 
   

# Equal axis scaling
x_limits = ax.get_xlim3d()
y_limits = ax.get_ylim3d()
z_limits = ax.get_zlim3d()

max_range = np.array([x_limits[1] - x_limits[0], y_limits[1] - y_limits[0], z_limits[1] - z_limits[0]]).max() / 4.0

mid_x = (x_limits[1] + x_limits[0]) * 0.5
mid_y = (y_limits[1] + y_limits[0]) * 0.5
mid_z = (z_limits[1] + z_limits[0]) * 0.5

ax.set_xlim3d([mid_x - max_range, mid_x + max_range])
ax.set_ylim3d([mid_y - max_range, mid_y + max_range])
ax.set_zlim3d([mid_z - max_range, mid_z + max_range])    


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend(['FHA', 'FHA Position', 'Sensor 1', 'Sensor 2'])

plt.show()
