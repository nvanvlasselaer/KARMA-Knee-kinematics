import os
from transforms3d import _gohlketransforms as t3d
import scipy.signal 
import numpy as np
from matplotlib import pyplot as plt

## uncomment if using executable

import sys
import ast


def Compute_Orthopedic_Metrics(subjects,conditions,Tests,sides,angle_maxs,point,input_folder=None):
    
    def save_Cardan_angles(angles, txtfilename,csvfilename,time):

        np.savetxt(txtfilename,angles)
        dataframe = pd.read_csv(txtfilename, delimiter=' ')
        dataframe.columns = ['F/E-axis','ADD/ABD-axis','INT/EXT-axis']
        dataframe['Motion %'] = time.tolist()

        # storing this dataframe in a csv file
        dataframe.to_csv(csvfilename, 
                          index = None)
    
# print(type(angle_maxs))
    for angle in angle_maxs:

        for Test in Tests:
            for condition in conditions:
#                 for movement in movements:

                print(subjects)
                # print(np.size(subjects))

                X21_Tf_all = np.zeros((1,100));Y21_Tf_all = np.zeros((1,100));Z21_Tf_all = np.zeros((1,100))
                X21_Tf_all_mean = np.zeros((100));Y21_Tf_all_mean = np.zeros((100));Z21_Tf_all_mean = np.zeros((100))
                X21_Tf_all_sd = np.zeros((100));Y21_Tf_all_sd = np.zeros((100));Z21_Tf_all_sd = np.zeros((100))

                X21_Pf_all = np.zeros((1,100));Y21_Pf_all = np.zeros((1,100));Z21_Pf_all = np.zeros((1,100))
                X21_Pf_all_mean = np.zeros((100));Y21_Pf_all_mean = np.zeros((100));Z21_Pf_all_mean = np.zeros((100))
                X21_Pf_all_sd = np.zeros((100));Y21_Pf_all_sd = np.zeros((100));Z21_Pf_all_sd = np.zeros((100))

                Femur_surface_point_patella_LCS_X_all = np.zeros((1,100));Femur_surface_point_patella_LCS_Y_all = np.zeros((1,100));Femur_surface_point_patella_LCS_Z_all = np.zeros((1,100))
                Patella_surface_point_LCS_X_all = np.zeros((1,100));Patella_surface_point_LCS_Y_all = np.zeros((1,100));Patella_surface_point_LCS_Z_all = np.zeros((1,100))
                Femur_lat_cond_surface_point_LCS_X_all = np.zeros((1,100));Femur_lat_cond_surface_point_LCS_Y_all = np.zeros((1,100));Femur_lat_cond_surface_point_LCS_Z_all = np.zeros((1,100))
                Femur_med_cond_surface_point_LCS_X_all = np.zeros((1,100));Femur_med_cond_surface_point_LCS_Y_all = np.zeros((1,100));Femur_med_cond_surface_point_LCS_Z_all = np.zeros((1,100))
                Tibia_lat_cond_surface_point_LCS_X_all = np.zeros((1,100));Tibia_lat_cond_surface_point_LCS_Y_all = np.zeros((1,100));Tibia_lat_cond_surface_point_LCS_Z_all = np.zeros((1,100))
                Tibia_med_cond_surface_point_LCS_X_all = np.zeros((1,100));Tibia_med_cond_surface_point_LCS_Y_all = np.zeros((1,100));Tibia_med_cond_surface_point_LCS_Z_all = np.zeros((1,100))

                Patella_translation_delta_X_all = np.zeros((1,100));Patella_translation_delta_X_all_mean = np.zeros((100));Patella_translation_delta_X_all_sd = np.zeros((100))
                Medial_cond_delta_X_all = np.zeros((1,100));Medial_cond_delta_X_all_mean = np.zeros((100));Medial_cond_delta_X_all_sd = np.zeros((100))
                Lateral_cond_delta_X_all = np.zeros((1,100));Lateral_cond_delta_X_all_mean = np.zeros((100));Lateral_cond_delta_X_all_sd = np.zeros((100))

                Patella_translation_delta_Y_all = np.zeros((1,100));Patella_translation_delta_Y_all_mean = np.zeros((100));Patella_translation_delta_Y_all_sd = np.zeros((100))
                Medial_cond_delta_Y_all = np.zeros((1,100));Medial_cond_delta_Y_all_mean = np.zeros((100));Medial_cond_delta_Y_all_sd = np.zeros((100))
                Lateral_cond_delta_Y_all = np.zeros((1,100));Lateral_cond_delta_Y_all_mean = np.zeros((100));Lateral_cond_delta_Y_all_sd = np.zeros((100))

                Patella_translation_delta_Z_all = np.zeros((1,100));Patella_translation_delta_Z_all_mean = np.zeros((100));Patella_translation_delta_Z_all_sd = np.zeros((100))
                Medial_cond_delta_Z_all = np.zeros((1,100));Medial_cond_delta_Z_all_mean = np.zeros((100));Medial_cond_delta_Z_all_sd = np.zeros((100))
                Lateral_cond_delta_Z_all = np.zeros((1,100));Lateral_cond_delta_Z_all_mean = np.zeros((100));Lateral_cond_delta_Z_all_sd = np.zeros((100))

                TTTG_d_all = np.zeros((1,100));TTTG_d_all_mean = np.zeros((100)); TTTG_d_all_sd = np.zeros((100))
                BO_perc_all = np.zeros((1,100));BO_perc_all_mean = np.zeros((100)); BO_perc_all_sd = np.zeros((100))
                alpha_all = np.zeros((1,100));alpha_all_mean = np.zeros((100)); alpha_all_sd = np.zeros((100))

                a = 0
                subjects_included = list()

                for side in sides:

                    print(side)

                    for subject in subjects:
                        
                        
                        if input_folder==None:
                            Dynamic_MSK_folder=f'/Input/{Test}/{subject}/{side}'
                            # Dynamic_MSK_folder=f'/Input/HS_025/Left/points/'
                        else:
                            Dynamic_MSK_folder=input_folder
                            
                        folder_with_results = f'{Dynamic_MSK_folder}/Metrics_results'
                        # directory = Directory_data + condition + '/Input/' + subject+ '/' + Test + '/' + side + '/' + movement + '/' + point + '/'
                        directory=f'{Dynamic_MSK_folder}/{point}/'
                        # directory=f'{Dynamic_MSK_folder}/{point}/'
                        print(directory)

                        if os.path.exists(directory) == False:
                            pass
                        else:                        

                            os.chdir(directory)

                            Femur = np.genfromtxt(directory + 'Femur_mypts.csv', dtype = 'float', delimiter = ',', filling_values = np.NaN)
                            Tibia = np.genfromtxt(directory + 'Tibia_mypts.csv', dtype = 'float', delimiter = ',', filling_values = np.NaN)
                            Patella = np.genfromtxt(directory + 'Patella_mypts.csv', dtype = 'float', delimiter = ',', filling_values = np.NaN)
                            
#                             print('len patella',Patella.shape)

                            # if Test == 'Test1':
                            header = 1
                            # else:
                                # header =0

                            points = np.genfromtxt(directory + 'points.txt', dtype = 'float', delimiter = ' ', skip_header = header, filling_values = np.NaN)

                            Femur = Femur[~np.isnan(Femur).any(axis=1)]
                            Tibia = Tibia[~np.isnan(Tibia).any(axis=1)]
                            Patella = Patella[~np.isnan(Patella).any(axis=1)]

                            Femur = np.delete(Femur, 1, 0)
                            Tibia = np.delete(Tibia, 1, 0)
                            Patella = np.delete(Patella, 1, 0)
                            
                            


                            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

                            ## Plot all points of the fix image  ###

                            # fig = plt.figure()
                            # ax = fig.gca(projection='3d')
                            # ax.scatter(points[0:9,0],points[0:9,1],points[0:9,2], color = 'r')
                            # ax.scatter(points[9:15,0],points[9:15,1],points[9:15,2], color = 'g')
                            # ax.scatter(points[15:22,0],points[15:22,1],points[15:22,2], color = 'b')
                            # plt.show()

                            # take the smallest number of time point. if registration fails earlier for one bone you can't do calculations with the others unless you know what joint you want to consider
                            time_points_all = [np.size(Tibia,0),np.size(Femur,0),np.size(Patella,0)]
                            time_points = np.min(time_points_all)
                            
                            
                            #time_points = 10


                            ## Rigit and left use the same code as the point collection is based on right left side of the imgage from
                            ## vv, therefore left side is lateral and right side is medial for the right leg BUT the other way around for left leg.
                            ## However, as the reference frame needs to point rightward, the i of the left reference frame needs to be in the opposit direction so
                            ##having lateral and medial inverted solve the problem

                            ### Coordinates all points of the fix image

                                ### FEMUR 

                            Femur_EpiM = Femur[0:time_points,0:3]
                            Femur_EpiL= Femur[0:time_points,3:6]
                            Femur_Center_diaf = Femur[0:time_points,6:9]
                            Femur_Post_EpiM = Femur[0:time_points,9:12]
                            Femur_Post_EpiL = Femur[0:time_points,12:15]
                            Femur_TGroove = Femur[0:time_points,15:18]
                            Femur_surface_point_patella_GCS = Femur[0:time_points,18:21]
                            Femur_med_cond_surface_point_GCS = Femur[0:time_points,21:24]
                            Femur_lat_cond_surface_point_GCS = Femur[0:time_points,24:27]
                            Femur_medial_TG = Femur[0:time_points,27:30]
                            Femur_lateral_TG = Femur[0:time_points,30:33]

                                ### TIBIA 

                            Tibia_ConM = Tibia[0:time_points,0:3]
                            Tibia_ConL = Tibia[0:time_points,3:6]
                            Tibia_Center_diaf= Tibia[0:time_points,6:9]
                            Tibia_med_cond_surface_point_GCS = Tibia[0:time_points,9:12]
                            Tibia_lat_cond_surface_point_GCS = Tibia[0:time_points,12:15]
                            Tibia_TT = Tibia[0:time_points,15:18]

                                ### PATELLA 

                            Patella_Ant = Patella[0:time_points,0:3]
                            Patella_Post = Patella[0:time_points,3:6]
                            Patella_Sup = Patella[0:time_points,6:9]
                            Patella_Inf = Patella[0:time_points,9:12]
                            Patella_Med = Patella[0:time_points,12:15]
                            Patella_Lat = Patella[0:time_points,15:18]
                            Patella_surface_point_GCS = Patella[0:time_points,18:21]
#                             print('Patella_Lat ',Patella[0:time_points,18:21] )
                         

                            #plt.figure()
                            #x = np.arange(0,np.size(Front_femur,0))
                            #plt.scatter(x,Front_femur[:,0])
                            #plt.scatter(x,Front_femur[:,1])
                            #plt.scatter(x,Front_femur[:,2])
                            #
                            #plt.figure()
                            #x = np.arange(0,np.size(TT_Tibia,0))
                            #plt.scatter(x,TT_Tibia[:,0])
                            #plt.scatter(x,TT_Tibia[:,1])
                            #plt.scatter(x,TT_Tibia[:,2])

                            ###############################################

                            Patella_surface_point_LCS = np.zeros((time_points,3))
                            Femur_surface_point_patella_LCS= np.zeros((time_points,3))

                            Femur_lat_cond_surface_point_LCS= np.zeros((time_points,3))
                            Femur_med_cond_surface_point_LCS= np.zeros((time_points,3))

                            Tibia_lat_cond_surface_point_LCS= np.zeros((time_points,3))
                            Tibia_med_cond_surface_point_LCS= np.zeros((time_points,3))
                            
                            
                            cardan_angles_Tf = np.zeros((100 ,3), dtype=float)
                            cardan_angles_Pf = np.zeros((100 ,3), dtype=float)


                            ###############################################

                            T = time_points
                            print(T)

                            Dfem_k = np.zeros((T,3)); Dfem_h = np.zeros((T,3)); Dfem_i = np.zeros((T,3)); Dfem_j = np.zeros((T,3))
                            Dynfem_R = np.zeros((3,3,T))

                            Dtib_k = np.zeros((T,3)); Dtib_h = np.zeros((T,3)); Dtib_i = np.zeros((T,3)); Dtib_j = np.zeros((T,3))
                            Dyntib_R = np.zeros((3,3,T))

                            Dpat_k = np.zeros((T,3)); Dpat_h = np.zeros((T,3)); Dpat_i = np.zeros((T,3)); Dpat_j = np.zeros((T,3))
                            Dynpat_R = np.zeros((3,3,T))

                            O_fem_dyn = np.zeros((T,3)); O_tib_dyn_lcs = np.zeros((T,3)); O_pat_dyn = np.zeros((T,3))

                            ROT_Fem_Tib = np.zeros((3,3,T))
                            ROT_Fem_pat = np.zeros((3,3,T))

                            X21_Tf = np.zeros((T)) ; Y21_Tf = np.zeros((T)) ; Z21_Tf = np.zeros((T))
                            X21_Pf = np.zeros((T)) ; Y21_Pf = np.zeros((T)) ; Z21_Pf = np.zeros((T))

                            TTTG_d = np.zeros((T)) ; BO_perc = np.zeros((T)) ; alpha = np.zeros((T))
                            Lateral_trochlear_inclination= np.zeros((T)); Sulcus_depth_d= np.zeros((T)); sulcus_angle= np.zeros((T))        
                            LPFa= np.zeros((T))

                            Tibia_ConM_lcs = np.zeros((T,3)); Tibia_ConL_lcs = np.zeros((T,3));Tibia_Center_diaf_lcs = np.zeros((T,3))
                            Patella_Med_lcs = np.zeros((T,3)); Patella_Lat_lcs = np.zeros((T,3));Patella_surface_point_GCS_lcs = np.zeros((T,3))

                            # femur_ax = np.zeros((T,np.size(x)))
                            # patella_ax = np.zeros((T,np.size(x)))

                            ### Calculate 3D kinematics - angles         

                            for t in np.arange(0,T):


                                ### Define reference frame on the femur ##

                                O_fem_dyn[t,:] = 0.5*(Femur_EpiL[t,:] + Femur_EpiM[t,:])

                                Dfem_h[t,:] = (Femur_Center_diaf[t,:] - O_fem_dyn[t,:])/np.linalg.norm(Femur_Center_diaf[t,:] - O_fem_dyn[t,:])
                                Dfem_i[t,:] = (Femur_EpiL[t,:] - O_fem_dyn[t,:])/np.linalg.norm(Femur_EpiL[t,:] - O_fem_dyn[t,:])
                                Dfem_j[t,:] = np.cross(Dfem_h[t,:], Dfem_i[t,:])
                                Dfem_k[t,:] = np.cross(Dfem_i[t,:], Dfem_j[t,:])
                                Dynfem_R[:,:,t] = np.c_[Dfem_i[t,:], Dfem_j[t,:], Dfem_k[t,:]]


                                ### transform all the coordinates in the new locel RF ##

                                 ### TIBIA 
                                Tibia_ConM_lcs[t,:] = (np.linalg.inv(Dynfem_R[:,:,t])).dot(Tibia_ConM[t,:] - O_fem_dyn[t,:])
                                Tibia_ConL_lcs[t,:] =(np.linalg.inv(Dynfem_R[:,:,t])).dot (Tibia_ConL[t,:] - O_fem_dyn[t,:])
                                Tibia_Center_diaf_lcs[t,:] = (np.linalg.inv(Dynfem_R[:,:,t])).dot(Tibia_Center_diaf[t,:] - O_fem_dyn[t,:])


                                ##Tibia
                                O_tib_dyn_lcs[t,:] = 0.5*(Tibia_ConL_lcs[t,:] + Tibia_ConM_lcs[t,:])
                                Dtib_h[t,:] = (Tibia_Center_diaf_lcs[t,:] - O_tib_dyn_lcs[t,:])/np.linalg.norm(Tibia_Center_diaf_lcs[t,:] - O_tib_dyn_lcs[t,:])
                                Dtib_i[t,:] = (Tibia_ConL_lcs[t,:] - O_tib_dyn_lcs[t,:])/np.linalg.norm(Tibia_ConL_lcs[t,:] - O_tib_dyn_lcs[t,:])
                                Dtib_j[t,:] = np.cross(Dtib_i[t,:], Dtib_h[t,:])
                                Dtib_k[t,:] = np.cross(Dtib_i[t,:], Dtib_j[t,:])
                                Dyntib_R[:,:,t] = np.c_[Dtib_i[t,:], Dtib_j[t,:], Dtib_k[t,:]]


                                 ### PATELLA
                                 # trans coo to the new RF                
                                Patella_Med_lcs[t,:] = (np.linalg.inv(Dynfem_R[:,:,t])).dot(Patella_Med[t,:] - O_fem_dyn[t,:])
                                Patella_Lat_lcs[t,:] = (np.linalg.inv(Dynfem_R[:,:,t])).dot(Patella_Lat[t,:] - O_fem_dyn[t,:])
                                
                                
                                Patella_surface_point_GCS_lcs[t,:] = (np.linalg.inv(Dynfem_R[:,:,t])).dot(Patella_surface_point_GCS[t,:] - O_fem_dyn[t,:])


                                O_pat_dyn[t,:] = 0.5*(Patella_Med_lcs[t,:] + Patella_Lat_lcs[t,:])
                                Dpat_h[t,:] = (Patella_surface_point_GCS_lcs[t,:] - O_pat_dyn[t,:])/np.linalg.norm(Patella_surface_point_GCS_lcs[t,:] - O_pat_dyn[t,:])
                                Dpat_i[t,:] = (Patella_Lat_lcs[t,:] - O_pat_dyn[t,:])/np.linalg.norm(Patella_Lat_lcs[t,:] - O_pat_dyn[t,:])
                                Dpat_k[t,:] = np.cross(Dpat_h[t,:], Dpat_i[t,:])
                                Dpat_j[t,:] = np.cross(Dpat_k[t,:], Dpat_i[t,:])
                                Dynpat_R[:,:,t] = np.c_[Dpat_i[t,:], Dpat_j[t,:], Dpat_k[t,:]]


                                ### ANGLES TIBIO_FEMORAL JOINT

                                if side == 'right':
                                    X21_Tf[t] = (t3d.euler_from_matrix(Dyntib_R[:,:,t], 'sxyz'))[0]*180/np.pi
                                    Y21_Tf[t] = (t3d.euler_from_matrix(Dyntib_R[:,:,t], 'sxyz'))[1]*180/np.pi
                                    Z21_Tf[t] = (t3d.euler_from_matrix(Dyntib_R[:,:,t], 'sxyz'))[2]*180/np.pi

                                else:
                                    X21_Tf[t] = (t3d.euler_from_matrix(Dyntib_R[:,:,t], 'sxyz'))[0]*180/np.pi
                                    Y21_Tf[t] = -(t3d.euler_from_matrix(Dyntib_R[:,:,t], 'sxyz'))[1]*180/np.pi
                                    Z21_Tf[t] = -(t3d.euler_from_matrix(Dyntib_R[:,:,t], 'sxyz'))[2]*180/np.pi

                                ### ANGLES PATELLO_FEMORAL JOINT
                                if side == 'right':

                                    X21_Pf[t] = (t3d.euler_from_matrix(Dynpat_R[:,:,t], 'sxyz'))[0]*180/np.pi
                                    Y21_Pf[t] = (t3d.euler_from_matrix(Dynpat_R[:,:,t], 'sxyz'))[1]*180/np.pi
                                    Z21_Pf[t] = (t3d.euler_from_matrix(Dynpat_R[:,:,t], 'sxyz'))[2]*180/np.pi
                                else:
                                    X21_Pf[t] = (t3d.euler_from_matrix(Dynpat_R[:,:,t], 'sxyz'))[0]*180/np.pi
                                    Y21_Pf[t] = -(t3d.euler_from_matrix(Dynpat_R[:,:,t], 'sxyz'))[1]*180/np.pi
                                    Z21_Pf[t] = -(t3d.euler_from_matrix(Dynpat_R[:,:,t], 'sxyz'))[2]*180/np.pi


                                

                                Patella_surface_point_LCS[t,:] = (np.linalg.inv(Dynfem_R[:,:,t])).dot(Patella_surface_point_GCS[t,:] - O_fem_dyn[t,:])
                                Femur_surface_point_patella_LCS[t,:] = (np.linalg.inv(Dynfem_R[:,:,t])).dot(Femur_surface_point_patella_GCS[t,:] - O_fem_dyn[t,:])

                                Femur_lat_cond_surface_point_LCS[t,:] = (np.linalg.inv(Dynfem_R[:,:,t])).dot(Femur_lat_cond_surface_point_GCS[t,:] - O_fem_dyn[t,:])
                                Femur_med_cond_surface_point_LCS[t,:] = (np.linalg.inv(Dynfem_R[:,:,t])).dot(Femur_med_cond_surface_point_GCS[t,:] - O_fem_dyn[t,:])

                                Tibia_lat_cond_surface_point_LCS[t,:] = (np.linalg.inv(Dynfem_R[:,:,t])).dot(Tibia_lat_cond_surface_point_GCS[t,:] - O_fem_dyn[t,:])
                                Tibia_med_cond_surface_point_LCS[t,:] = (np.linalg.inv(Dynfem_R[:,:,t])).dot(Tibia_med_cond_surface_point_GCS[t,:] - O_fem_dyn[t,:])



                        ###########################################################################################################################
                        ### MORPHOLOGICAL PARAMETERS
                        ###########################################################################################################################

                                if side == 'right': 

                                    A = Femur_Post_EpiM[t,0:2]
                                    B = Femur_Post_EpiL[t,0:2]
                                    C = Femur_TGroove[t,0:2]
                                    D = Tibia_TT[t,0:2]                                
                                    E = Patella_Med[t,0:2]
                                    F = Patella_Lat[t,0:2]

                                    L = Femur_medial_TG[t,0:2]
                                    M = Femur_lateral_TG[t,0:2]

                                    P = Patella_Post[t,0:2]

                                else:

                                    B = Femur_Post_EpiM[t,0:2]
                                    A = Femur_Post_EpiL[t,0:2]
                                    C = Femur_TGroove[t,0:2]
                                    D = Tibia_TT[t,0:2]
                                    F = Patella_Med[t,0:2]
                                    E = Patella_Lat[t,0:2]

                                    L = Femur_lateral_TG[t,0:2]
                                    M = Femur_medial_TG[t,0:2]

                                    P = Patella_Post[t,0:2]

                                ## y = a*x + b

                                ## line c passing through A and B

                                a_c = (B[1]-A[1])/(B[0]-A[0])
                                b_c = B[1] - a_c*B[0]
                                #c_coeff = line(A, B)
                                #a_c = c_coeff[0]
                                #b_c =c_coeff[1]

                                ## line tg passing trhough C perpendicular to line c

                                if a_c == 0:

                                    TTTG_d[t] = C[0] - D[0]

                                else:

                                    a_tg = -(1/a_c)
                                    b_tg = C[1] + (1/a_c)*C[0]

                                    ## intersection between tg and c
                                    xi_tg_c = (b_tg - b_c) / (a_tg + a_c)
                                    yi_tg_c = a_c * xi_tg_c + b_c
                                    G = (xi_tg_c,yi_tg_c)


                                    ## line tt passing trhough D perpendicular to line c
                                    a_tt = -(1/a_c)
                                    b_tt = D[1] + (1/a_c)*D[0]

                                    ## intersection between tt and c
                                    xi_tt_c = ( b_tt - b_c) / (a_tt + a_c)
                                    yi_tt_c = a_c * xi_tt_c + b_c
                                    H= (xi_tt_c,yi_tt_c)

                                    ### TTTG DISTANCE ###


                                    TTTG_d[t] = np.sqrt( ((G[0]-H[0])**2)+((G[1]-H[1])**2) )


                                    ### Bisec Ofsset ###

                                ## line p passing through E and F    
                                a_p = (F[1]-E[1])/(F[0]-E[0])
                                b_p = F[1] - a_p*F[0]

                                 ## intersection between tg and p
                                if a_c == 0:
                                    # Line 'c' is horizontal, so line 'tg' is vertical
                                    xi_tg = C[0]  # x-coordinate of line 'tg'
                                    yi_tg_p = a_p * xi_tg + b_p  # Calculate y-coordinate where line 'tg' intersects line 'p'
                                    I = (xi_tg, yi_tg_p)  # Intersection point between line 'tg' and line 'p'
                                else:
                                    ## intersection between tg and p
                                    xi_tg_p = (b_p - b_tg) / (a_tg - a_p)
                                    # xi_tg_p = ( b_tg - b_p) / (a_p + a_tg)
                                    yi_tg_p = a_p * xi_tg_p + b_p
                                    I = (xi_tg_p, yi_tg_p)

                                IF_dist = np.sqrt( ((I[0]-F[0])**2)+((I[1]-F[1])**2) )

                                EF_dist = np.sqrt( ((E[0]-F[0])**2)+((E[1]-F[1])**2) )

                                BO_perc[t] = (IF_dist/(EF_dist))*100


                                    ### Lateral Patellar TILT ###

                                if side == 'right': 
                                    alpha[t] = -(np.arctan((a_p - a_c)/(1+ (a_p*a_c ))))*180/np.pi

                                    # alpha[t] = (np.arctan2((E[1]-F[1]),E[0]-F[0]))-(np.arctan2((A[1]-B[1]),A[0]-B[0]))*180/np.pi  
                                else:
                                    alpha[t] = (np.arctan((a_p - a_c)/(1+ (a_p*a_c ))))*180/np.pi      

                            ##Interpolation 

                            X21_Tf_norm = np.interp(x = np.linspace(0,np.size(X21_Tf),num=100), xp = np.linspace(0,np.size(X21_Tf),num = np.size(X21_Tf)), fp = X21_Tf)
                            Y21_Tf_norm = np.interp(x = np.linspace(0,np.size(Y21_Tf),num=100), xp = np.linspace(0,np.size(Y21_Tf),num = np.size(Y21_Tf)), fp = Y21_Tf)
                            Z21_Tf_norm = np.interp(x = np.linspace(0,np.size(Z21_Tf),num=100), xp = np.linspace(0,np.size(Z21_Tf),num = np.size(Z21_Tf)), fp = Z21_Tf)

                            X21_Pf_norm = np.interp(x = np.linspace(0,np.size(X21_Pf),num=100), xp = np.linspace(0,np.size(X21_Pf),num = np.size(X21_Pf)), fp = X21_Pf)
                            Y21_Pf_norm = np.interp(x = np.linspace(0,np.size(Y21_Pf),num=100), xp = np.linspace(0,np.size(Y21_Pf),num = np.size(Y21_Pf)), fp = Y21_Pf)
                            Z21_Pf_norm = np.interp(x = np.linspace(0,np.size(Z21_Pf),num=100), xp = np.linspace(0,np.size(Z21_Pf),num = np.size(Z21_Pf)), fp = Z21_Pf)

                            Patella_surface_point_LCS_norm_X = np.interp(x = np.linspace(0,np.size(Patella_surface_point_LCS[:,0]),num=100), xp = np.linspace(0,np.size(Patella_surface_point_LCS[:,0]),num = np.size(Patella_surface_point_LCS[:,0])), fp = Patella_surface_point_LCS[:,0])
                            Patella_surface_point_LCS_norm_Y = np.interp(x = np.linspace(0,np.size(Patella_surface_point_LCS[:,1]),num=100), xp = np.linspace(0,np.size(Patella_surface_point_LCS[:,1]),num = np.size(Patella_surface_point_LCS[:,1])), fp = Patella_surface_point_LCS[:,1])
                            Patella_surface_point_LCS_norm_Z = np.interp(x = np.linspace(0,np.size(Patella_surface_point_LCS[:,2]),num=100), xp = np.linspace(0,np.size(Patella_surface_point_LCS[:,2]),num = np.size(Patella_surface_point_LCS[:,2])), fp = Patella_surface_point_LCS[:,2])

                            Femur_surface_point_patella_LCS_norm_X = np.interp(x = np.linspace(0,np.size(Femur_surface_point_patella_LCS[:,0]),num=100), xp = np.linspace(0,np.size(Femur_surface_point_patella_LCS[:,0]),num = np.size(Femur_surface_point_patella_LCS[:,0])), fp = Femur_surface_point_patella_LCS[:,0])
                            Femur_surface_point_patella_LCS_norm_Y = np.interp(x = np.linspace(0,np.size(Femur_surface_point_patella_LCS[:,1]),num=100), xp = np.linspace(0,np.size(Femur_surface_point_patella_LCS[:,1]),num = np.size(Femur_surface_point_patella_LCS[:,1])), fp = Femur_surface_point_patella_LCS[:,1])
                            Femur_surface_point_patella_LCS_norm_Z = np.interp(x = np.linspace(0,np.size(Femur_surface_point_patella_LCS[:,2]),num=100), xp = np.linspace(0,np.size(Femur_surface_point_patella_LCS[:,2]),num = np.size(Femur_surface_point_patella_LCS[:,2])), fp = Femur_surface_point_patella_LCS[:,2])

                            Femur_lat_cond_surface_point_LCS_norm_X = np.interp(x = np.linspace(0,np.size(Femur_lat_cond_surface_point_LCS[:,0]),num=100), xp = np.linspace(0,np.size(Femur_lat_cond_surface_point_LCS[:,0]),num = np.size(Femur_lat_cond_surface_point_LCS[:,0])), fp = Femur_lat_cond_surface_point_LCS[:,0])
                            Femur_lat_cond_surface_point_LCS_norm_Y = np.interp(x = np.linspace(0,np.size(Femur_lat_cond_surface_point_LCS[:,1]),num=100), xp = np.linspace(0,np.size(Femur_lat_cond_surface_point_LCS[:,1]),num = np.size(Femur_lat_cond_surface_point_LCS[:,1])), fp = Femur_lat_cond_surface_point_LCS[:,1])
                            Femur_lat_cond_surface_point_LCS_norm_Z = np.interp(x = np.linspace(0,np.size(Femur_lat_cond_surface_point_LCS[:,2]),num=100), xp = np.linspace(0,np.size(Femur_lat_cond_surface_point_LCS[:,2]),num = np.size(Femur_lat_cond_surface_point_LCS[:,2])), fp = Femur_lat_cond_surface_point_LCS[:,2])

                            Femur_med_cond_surface_point_LCS_norm_X = np.interp(x = np.linspace(0,np.size(Femur_med_cond_surface_point_LCS[:,0]),num=100), xp = np.linspace(0,np.size(Femur_med_cond_surface_point_LCS[:,0]),num = np.size(Femur_med_cond_surface_point_LCS[:,0])), fp = Femur_med_cond_surface_point_LCS[:,0])
                            Femur_med_cond_surface_point_LCS_norm_Y = np.interp(x = np.linspace(0,np.size(Femur_med_cond_surface_point_LCS[:,1]),num=100), xp = np.linspace(0,np.size(Femur_med_cond_surface_point_LCS[:,1]),num = np.size(Femur_med_cond_surface_point_LCS[:,1])), fp = Femur_med_cond_surface_point_LCS[:,1])
                            Femur_med_cond_surface_point_LCS_norm_Z = np.interp(x = np.linspace(0,np.size(Femur_med_cond_surface_point_LCS[:,2]),num=100), xp = np.linspace(0,np.size(Femur_med_cond_surface_point_LCS[:,2]),num = np.size(Femur_med_cond_surface_point_LCS[:,2])), fp = Femur_med_cond_surface_point_LCS[:,2])

                            Tibia_lat_cond_surface_point_LCS_norm_X = np.interp(x = np.linspace(0,np.size(Tibia_lat_cond_surface_point_LCS[:,0]),num=100), xp = np.linspace(0,np.size(Tibia_lat_cond_surface_point_LCS[:,0]),num = np.size(Tibia_lat_cond_surface_point_LCS[:,0])), fp = Tibia_lat_cond_surface_point_LCS[:,0])
                            Tibia_lat_cond_surface_point_LCS_norm_Y = np.interp(x = np.linspace(0,np.size(Tibia_lat_cond_surface_point_LCS[:,1]),num=100), xp = np.linspace(0,np.size(Tibia_lat_cond_surface_point_LCS[:,1]),num = np.size(Tibia_lat_cond_surface_point_LCS[:,1])), fp = Tibia_lat_cond_surface_point_LCS[:,1])
                            Tibia_lat_cond_surface_point_LCS_norm_Z = np.interp(x = np.linspace(0,np.size(Tibia_lat_cond_surface_point_LCS[:,2]),num=100), xp = np.linspace(0,np.size(Tibia_lat_cond_surface_point_LCS[:,2]),num = np.size(Tibia_lat_cond_surface_point_LCS[:,2])), fp = Tibia_lat_cond_surface_point_LCS[:,2])

                            Tibia_med_cond_surface_point_LCS_norm_X = np.interp(x = np.linspace(0,np.size(Tibia_med_cond_surface_point_LCS[:,0]),num=100), xp = np.linspace(0,np.size(Tibia_med_cond_surface_point_LCS[:,0]),num = np.size(Tibia_med_cond_surface_point_LCS[:,0])), fp = Tibia_med_cond_surface_point_LCS[:,0])
                            Tibia_med_cond_surface_point_LCS_norm_Y = np.interp(x = np.linspace(0,np.size(Tibia_med_cond_surface_point_LCS[:,1]),num=100), xp = np.linspace(0,np.size(Tibia_med_cond_surface_point_LCS[:,1]),num = np.size(Tibia_med_cond_surface_point_LCS[:,1])), fp = Tibia_med_cond_surface_point_LCS[:,1])
                            Tibia_med_cond_surface_point_LCS_norm_Z = np.interp(x = np.linspace(0,np.size(Tibia_med_cond_surface_point_LCS[:,2]),num=100), xp = np.linspace(0,np.size(Tibia_med_cond_surface_point_LCS[:,2]),num = np.size(Tibia_med_cond_surface_point_LCS[:,2])), fp = Tibia_med_cond_surface_point_LCS[:,2])

                            TTTG_d_norm = np.interp(x = np.linspace(0,np.size(TTTG_d),num=100), xp = np.linspace(0,np.size(TTTG_d),num = np.size(TTTG_d)), fp = TTTG_d)
                            BO_perc_norm = np.interp(x = np.linspace(0,np.size(BO_perc),num=100), xp = np.linspace(0,np.size(BO_perc),num = np.size(BO_perc)), fp = BO_perc)
                            alpha_norm = np.interp(x = np.linspace(0,np.size(alpha),num=100), xp = np.linspace(0,np.size(alpha),num = np.size(alpha)), fp = alpha)


                            window = 51

                            X21_Tf_norm = scipy.signal.savgol_filter(X21_Tf_norm, window, 4)
                            Y21_Tf_norm = scipy.signal.savgol_filter(Y21_Tf_norm, window, 4)
                            Z21_Tf_norm = scipy.signal.savgol_filter(Z21_Tf_norm, window, 4)

                            X21_Pf_norm = scipy.signal.savgol_filter(X21_Pf_norm, window, 4)
                            Y21_Pf_norm = scipy.signal.savgol_filter(Y21_Pf_norm, window, 4)
                            Z21_Pf_norm = scipy.signal.savgol_filter(Z21_Pf_norm, window, 4)

                            Patella_surface_point_LCS_norm_X = scipy.signal.savgol_filter(Patella_surface_point_LCS_norm_X, window, 4)
                            Patella_surface_point_LCS_norm_Y = scipy.signal.savgol_filter(Patella_surface_point_LCS_norm_Y, window, 4)
                            Patella_surface_point_LCS_norm_Z = scipy.signal.savgol_filter(Patella_surface_point_LCS_norm_Z, window, 4)

                            Femur_surface_point_patella_LCS_norm_X = scipy.signal.savgol_filter(Femur_surface_point_patella_LCS_norm_X, window, 4)
                            Femur_surface_point_patella_LCS_norm_Y = scipy.signal.savgol_filter(Femur_surface_point_patella_LCS_norm_Y, window, 4)
                            Femur_surface_point_patella_LCS_norm_Z = scipy.signal.savgol_filter(Femur_surface_point_patella_LCS_norm_Z, window, 4)

                            Femur_lat_cond_surface_point_LCS_norm_X = scipy.signal.savgol_filter(Femur_lat_cond_surface_point_LCS_norm_X, window, 4)
                            Femur_lat_cond_surface_point_LCS_norm_Y = scipy.signal.savgol_filter(Femur_lat_cond_surface_point_LCS_norm_Y, window, 4)
                            Femur_lat_cond_surface_point_LCS_norm_Z = scipy.signal.savgol_filter(Femur_lat_cond_surface_point_LCS_norm_Z, window, 4)

                            Femur_med_cond_surface_point_LCS_norm_X = scipy.signal.savgol_filter(Femur_med_cond_surface_point_LCS_norm_X, window, 4)
                            Femur_med_cond_surface_point_LCS_norm_Y = scipy.signal.savgol_filter(Femur_med_cond_surface_point_LCS_norm_Y, window, 4)
                            Femur_med_cond_surface_point_LCS_norm_Z = scipy.signal.savgol_filter(Femur_med_cond_surface_point_LCS_norm_Z, window, 4)

                            Tibia_lat_cond_surface_point_LCS_norm_X = scipy.signal.savgol_filter(Tibia_lat_cond_surface_point_LCS_norm_X, window, 4)
                            Tibia_lat_cond_surface_point_LCS_norm_Y = scipy.signal.savgol_filter(Tibia_lat_cond_surface_point_LCS_norm_Y, window, 4)
                            Tibia_lat_cond_surface_point_LCS_norm_Z = scipy.signal.savgol_filter(Tibia_lat_cond_surface_point_LCS_norm_Z, window, 4)

                            Tibia_med_cond_surface_point_LCS_norm_X = scipy.signal.savgol_filter(Tibia_med_cond_surface_point_LCS_norm_X, window, 4)
                            Tibia_med_cond_surface_point_LCS_norm_Y = scipy.signal.savgol_filter(Tibia_med_cond_surface_point_LCS_norm_Y, window, 4)
                            Tibia_med_cond_surface_point_LCS_norm_Z = scipy.signal.savgol_filter(Tibia_med_cond_surface_point_LCS_norm_Z, window, 4)


                            TTTG_d_norm = scipy.signal.savgol_filter(TTTG_d_norm, window, 4)
                            BO_perc_norm = scipy.signal.savgol_filter(BO_perc_norm, window, 4)
                            alpha_norm = scipy.signal.savgol_filter(alpha_norm, window, 4)

                            ## Displace everything to zero

                            X21_Tf_norm = X21_Tf_norm - X21_Tf_norm[0]
                            Y21_Tf_norm = Y21_Tf_norm - Y21_Tf_norm[0]
                            Z21_Tf_norm = Z21_Tf_norm - Z21_Tf_norm[0]

                            X21_Pf_norm = X21_Pf_norm - X21_Pf_norm[0]
                            Y21_Pf_norm = Y21_Pf_norm - Y21_Pf_norm[0]
                            Z21_Pf_norm = Z21_Pf_norm - Z21_Pf_norm[0]

                            Patella_surface_point_LCS_norm_X = Patella_surface_point_LCS_norm_X - Patella_surface_point_LCS_norm_X[0]
                            Femur_surface_point_patella_LCS_norm_X = Femur_surface_point_patella_LCS_norm_X - Femur_surface_point_patella_LCS_norm_X[0]
                            Femur_lat_cond_surface_point_LCS_norm_X = Femur_lat_cond_surface_point_LCS_norm_X - Femur_lat_cond_surface_point_LCS_norm_X[0]
                            Femur_med_cond_surface_point_LCS_norm_X = Femur_med_cond_surface_point_LCS_norm_X - Femur_med_cond_surface_point_LCS_norm_X[0]
                            Tibia_lat_cond_surface_point_LCS_norm_X = Tibia_lat_cond_surface_point_LCS_norm_X - Tibia_lat_cond_surface_point_LCS_norm_X[0]
                            Tibia_med_cond_surface_point_LCS_norm_X = Tibia_med_cond_surface_point_LCS_norm_X - Tibia_med_cond_surface_point_LCS_norm_X[0]

                            Patella_surface_point_LCS_norm_Y = Patella_surface_point_LCS_norm_Y - Patella_surface_point_LCS_norm_Y[0]
                            Femur_surface_point_patella_LCS_norm_Y = Femur_surface_point_patella_LCS_norm_Y - Femur_surface_point_patella_LCS_norm_Y[0]
                            Femur_lat_cond_surface_point_LCS_norm_Y = Femur_lat_cond_surface_point_LCS_norm_Y - Femur_lat_cond_surface_point_LCS_norm_Y[0]
                            Femur_med_cond_surface_point_LCS_norm_Y = Femur_med_cond_surface_point_LCS_norm_Y - Femur_med_cond_surface_point_LCS_norm_Y[0]
                            Tibia_lat_cond_surface_point_LCS_norm_Y = Tibia_lat_cond_surface_point_LCS_norm_Y - Tibia_lat_cond_surface_point_LCS_norm_Y[0]
                            Tibia_med_cond_surface_point_LCS_norm_Y = Tibia_med_cond_surface_point_LCS_norm_Y - Tibia_med_cond_surface_point_LCS_norm_Y[0]

                            Patella_surface_point_LCS_norm_Z = Patella_surface_point_LCS_norm_Z - Patella_surface_point_LCS_norm_Z[0]
                            Femur_surface_point_patella_LCS_norm_Z = Femur_surface_point_patella_LCS_norm_Z - Femur_surface_point_patella_LCS_norm_Z[0]
                            Femur_lat_cond_surface_point_LCS_norm_Z = Femur_lat_cond_surface_point_LCS_norm_Z - Femur_lat_cond_surface_point_LCS_norm_Z[0]
                            Femur_med_cond_surface_point_LCS_norm_Z = Femur_med_cond_surface_point_LCS_norm_Z - Femur_med_cond_surface_point_LCS_norm_Z[0]
                            Tibia_lat_cond_surface_point_LCS_norm_Z = Tibia_lat_cond_surface_point_LCS_norm_Z - Tibia_lat_cond_surface_point_LCS_norm_Z[0]
                            Tibia_med_cond_surface_point_LCS_norm_Z = Tibia_med_cond_surface_point_LCS_norm_Z - Tibia_med_cond_surface_point_LCS_norm_Z[0]


                            ### Choose the angle at which do the analysis ###

                            for i in np.arange(0,100):
                                if X21_Tf_norm[i] > angle:
                                    pass
                                else:
                                    print(i)
                                    # i=100

                                                                        #subjects_temp = int(subject.split('_')[1])
                                    subjects_included = np.append(subjects_included,subject)

                                    X21_Tf_norm_cut = X21_Tf_norm[0:i]
                                    Y21_Tf_norm_cut = Y21_Tf_norm[0:i]
                                    Z21_Tf_norm_cut = Z21_Tf_norm[0:i]

                                    X21_Pf_norm_cut = X21_Pf_norm[0:i]
                                    Y21_Pf_norm_cut = Y21_Pf_norm[0:i]
                                    Z21_Pf_norm_cut = Z21_Pf_norm[0:i]


                                    Femur_surface_point_patella_LCS_norm_X_cut = Femur_surface_point_patella_LCS_norm_X[0:i]
                                    Femur_surface_point_patella_LCS_norm_Y_cut = Femur_surface_point_patella_LCS_norm_Y[0:i]
                                    Femur_surface_point_patella_LCS_norm_Z_cut = Femur_surface_point_patella_LCS_norm_Z[0:i]

                                    Femur_lat_cond_surface_point_LCS_norm_X_cut = Femur_lat_cond_surface_point_LCS_norm_X[0:i]
                                    Femur_lat_cond_surface_point_LCS_norm_Y_cut = Femur_lat_cond_surface_point_LCS_norm_Y[0:i]
                                    Femur_lat_cond_surface_point_LCS_norm_Z_cut = Femur_lat_cond_surface_point_LCS_norm_Z[0:i]

                                    Femur_med_cond_surface_point_LCS_norm_X_cut = Femur_med_cond_surface_point_LCS_norm_X[0:i]
                                    Femur_med_cond_surface_point_LCS_norm_Y_cut = Femur_med_cond_surface_point_LCS_norm_Y[0:i]
                                    Femur_med_cond_surface_point_LCS_norm_Z_cut = Femur_med_cond_surface_point_LCS_norm_Z[0:i]

                                    if side == 'right': 

                                        Patella_surface_point_LCS_norm_X_cut = Patella_surface_point_LCS_norm_X[0:i]
                                        Tibia_lat_cond_surface_point_LCS_norm_X_cut = Tibia_lat_cond_surface_point_LCS_norm_X[0:i]
                                        Tibia_med_cond_surface_point_LCS_norm_X_cut = Tibia_med_cond_surface_point_LCS_norm_X[0:i]

                                    else:
                                        Patella_surface_point_LCS_norm_X_cut = -Patella_surface_point_LCS_norm_X[0:i]
                                        Tibia_lat_cond_surface_point_LCS_norm_X_cut = -Tibia_lat_cond_surface_point_LCS_norm_X[0:i]
                                        Tibia_med_cond_surface_point_LCS_norm_X_cut = -Tibia_med_cond_surface_point_LCS_norm_X[0:i]


                                    Patella_surface_point_LCS_norm_Y_cut = Patella_surface_point_LCS_norm_Y[0:i]
                                    Patella_surface_point_LCS_norm_Z_cut = Patella_surface_point_LCS_norm_Z[0:i]

                                    Tibia_lat_cond_surface_point_LCS_norm_Y_cut = Tibia_lat_cond_surface_point_LCS_norm_Y[0:i]
                                    Tibia_lat_cond_surface_point_LCS_norm_Z_cut = Tibia_lat_cond_surface_point_LCS_norm_Z[0:i]

                                    Tibia_med_cond_surface_point_LCS_norm_Y_cut = Tibia_med_cond_surface_point_LCS_norm_Y[0:i]
                                    Tibia_med_cond_surface_point_LCS_norm_Z_cut = Tibia_med_cond_surface_point_LCS_norm_Z[0:i]


                                    TTTG_d_norm_cut = TTTG_d_norm[0:i]
                                    BO_perc_norm_cut = BO_perc_norm[0:i]
                                    alpha_norm_cut = alpha_norm[0:i]

                                    #############################################################


                                    X21_Tf_norm_cut_int = np.interp(x = np.linspace(0,np.size(X21_Tf_norm_cut),num=100), xp = np.linspace(0,np.size(X21_Tf_norm_cut),num = np.size(X21_Tf_norm_cut)), fp = X21_Tf_norm_cut)
                                    Y21_Tf_norm_cut_int = np.interp(x = np.linspace(0,np.size(Y21_Tf_norm_cut),num=100), xp = np.linspace(0,np.size(Y21_Tf_norm_cut),num = np.size(Y21_Tf_norm_cut)), fp = Y21_Tf_norm_cut)
                                    Z21_Tf_norm_cut_int = np.interp(x = np.linspace(0,np.size(Z21_Tf_norm_cut),num=100), xp = np.linspace(0,np.size(Z21_Tf_norm_cut),num = np.size(Z21_Tf_norm_cut)), fp = Z21_Tf_norm_cut)

                                    X21_Tf_all = np.vstack((X21_Tf_all,X21_Tf_norm_cut_int))
                                    Y21_Tf_all = np.vstack((Y21_Tf_all,Y21_Tf_norm_cut_int));
                                    Z21_Tf_all = np.vstack((Z21_Tf_all,Z21_Tf_norm_cut_int));

                                    X21_Pf_norm_cut_int = np.interp(x = np.linspace(0,np.size(X21_Pf_norm_cut),num=100), xp = np.linspace(0,np.size(X21_Pf_norm_cut),num = np.size(X21_Pf_norm_cut)), fp = X21_Pf_norm_cut)
                                    Y21_Pf_norm_cut_int = np.interp(x = np.linspace(0,np.size(Y21_Pf_norm_cut),num=100), xp = np.linspace(0,np.size(Y21_Pf_norm_cut),num = np.size(Y21_Pf_norm_cut)), fp = Y21_Pf_norm_cut)
                                    Z21_Pf_norm_cut_int = np.interp(x = np.linspace(0,np.size(Z21_Pf_norm_cut),num=100), xp = np.linspace(0,np.size(Z21_Pf_norm_cut),num = np.size(Z21_Pf_norm_cut)), fp = Z21_Pf_norm_cut)

                                    X21_Pf_all = np.vstack((X21_Pf_all,X21_Pf_norm_cut_int))
                                    Y21_Pf_all = np.vstack((Y21_Pf_all,Y21_Pf_norm_cut_int));
                                    Z21_Pf_all = np.vstack((Z21_Pf_all,Z21_Pf_norm_cut_int));



                                    Femur_surface_point_patella_LCS_norm_X_cut_int = np.interp(x = np.linspace(0,np.size(Femur_surface_point_patella_LCS_norm_X_cut),num=100), xp = np.linspace(0,np.size(Femur_surface_point_patella_LCS_norm_X_cut),num = np.size(Femur_surface_point_patella_LCS_norm_X_cut)), fp = Femur_surface_point_patella_LCS_norm_X_cut)
                                    Femur_surface_point_patella_LCS_norm_Y_cut_int = np.interp(x = np.linspace(0,np.size(Femur_surface_point_patella_LCS_norm_Y_cut),num=100), xp = np.linspace(0,np.size(Femur_surface_point_patella_LCS_norm_Y_cut),num = np.size(Femur_surface_point_patella_LCS_norm_Y_cut)), fp = Femur_surface_point_patella_LCS_norm_Y_cut)
                                    Femur_surface_point_patella_LCS_norm_Z_cut_int = np.interp(x = np.linspace(0,np.size(Femur_surface_point_patella_LCS_norm_Z_cut),num=100), xp = np.linspace(0,np.size(Femur_surface_point_patella_LCS_norm_Z_cut),num = np.size(Femur_surface_point_patella_LCS_norm_Z_cut)), fp = Femur_surface_point_patella_LCS_norm_Z_cut)

                                    Femur_surface_point_patella_LCS_X_all = np.vstack((Femur_surface_point_patella_LCS_X_all,Femur_surface_point_patella_LCS_norm_X_cut_int))
                                    Femur_surface_point_patella_LCS_Y_all = np.vstack((Femur_surface_point_patella_LCS_Y_all,Femur_surface_point_patella_LCS_norm_Y_cut_int))
                                    Femur_surface_point_patella_LCS_Z_all = np.vstack((Femur_surface_point_patella_LCS_Z_all,Femur_surface_point_patella_LCS_norm_Z_cut_int))

                                    Patella_surface_point_LCS_norm_X_cut_int = np.interp(x = np.linspace(0,np.size(Patella_surface_point_LCS_norm_X_cut),num=100), xp = np.linspace(0,np.size(Patella_surface_point_LCS_norm_X_cut),num = np.size(Patella_surface_point_LCS_norm_X_cut)), fp = Patella_surface_point_LCS_norm_X_cut)
                                    Patella_surface_point_LCS_norm_Y_cut_int = np.interp(x = np.linspace(0,np.size(Patella_surface_point_LCS_norm_Y_cut),num=100), xp = np.linspace(0,np.size(Patella_surface_point_LCS_norm_Y_cut),num = np.size(Patella_surface_point_LCS_norm_Y_cut)), fp = Patella_surface_point_LCS_norm_Y_cut)
                                    Patella_surface_point_LCS_norm_Z_cut_int = np.interp(x = np.linspace(0,np.size(Patella_surface_point_LCS_norm_Z_cut),num=100), xp = np.linspace(0,np.size(Patella_surface_point_LCS_norm_Z_cut),num = np.size(Patella_surface_point_LCS_norm_Z_cut)), fp = Patella_surface_point_LCS_norm_Z_cut)

                                    Patella_surface_point_LCS_X_all = np.vstack((Patella_surface_point_LCS_X_all,Patella_surface_point_LCS_norm_X_cut_int))
                                    Patella_surface_point_LCS_Y_all = np.vstack((Patella_surface_point_LCS_Y_all,Patella_surface_point_LCS_norm_Y_cut_int))
                                    Patella_surface_point_LCS_Z_all = np.vstack((Patella_surface_point_LCS_Z_all,Patella_surface_point_LCS_norm_Z_cut_int))

                                    Femur_lat_cond_surface_point_LCS_norm_X_cut_int = np.interp(x = np.linspace(0,np.size(Femur_lat_cond_surface_point_LCS_norm_X_cut),num=100), xp = np.linspace(0,np.size(Femur_lat_cond_surface_point_LCS_norm_X_cut),num = np.size(Femur_lat_cond_surface_point_LCS_norm_X_cut)), fp = Femur_lat_cond_surface_point_LCS_norm_X_cut)
                                    Femur_lat_cond_surface_point_LCS_norm_Y_cut_int = np.interp(x = np.linspace(0,np.size(Femur_lat_cond_surface_point_LCS_norm_Y_cut),num=100), xp = np.linspace(0,np.size(Femur_lat_cond_surface_point_LCS_norm_Y_cut),num = np.size(Femur_lat_cond_surface_point_LCS_norm_Y_cut)), fp = Femur_lat_cond_surface_point_LCS_norm_Y_cut)
                                    Femur_lat_cond_surface_point_LCS_norm_Z_cut_int = np.interp(x = np.linspace(0,np.size(Femur_lat_cond_surface_point_LCS_norm_Z_cut),num=100), xp = np.linspace(0,np.size(Femur_lat_cond_surface_point_LCS_norm_Z_cut),num = np.size(Femur_lat_cond_surface_point_LCS_norm_Z_cut)), fp = Femur_lat_cond_surface_point_LCS_norm_Z_cut)

                                    Femur_lat_cond_surface_point_LCS_X_all = np.vstack((Femur_lat_cond_surface_point_LCS_X_all,Femur_lat_cond_surface_point_LCS_norm_X_cut_int))
                                    Femur_lat_cond_surface_point_LCS_Y_all = np.vstack((Femur_lat_cond_surface_point_LCS_Y_all,Femur_lat_cond_surface_point_LCS_norm_Y_cut_int))
                                    Femur_lat_cond_surface_point_LCS_Z_all = np.vstack((Femur_lat_cond_surface_point_LCS_Z_all,Femur_lat_cond_surface_point_LCS_norm_Z_cut_int))

                                    Femur_med_cond_surface_point_LCS_norm_X_cut_int = np.interp(x = np.linspace(0,np.size(Femur_med_cond_surface_point_LCS_norm_X_cut),num=100), xp = np.linspace(0,np.size(Femur_med_cond_surface_point_LCS_norm_X_cut),num = np.size(Femur_med_cond_surface_point_LCS_norm_X_cut)), fp = Femur_med_cond_surface_point_LCS_norm_X_cut)
                                    Femur_med_cond_surface_point_LCS_norm_Y_cut_int = np.interp(x = np.linspace(0,np.size(Femur_med_cond_surface_point_LCS_norm_Y_cut),num=100), xp = np.linspace(0,np.size(Femur_med_cond_surface_point_LCS_norm_Y_cut),num = np.size(Femur_med_cond_surface_point_LCS_norm_Y_cut)), fp = Femur_med_cond_surface_point_LCS_norm_Y_cut)
                                    Femur_med_cond_surface_point_LCS_norm_Z_cut_int = np.interp(x = np.linspace(0,np.size(Femur_med_cond_surface_point_LCS_norm_Z_cut),num=100), xp = np.linspace(0,np.size(Femur_med_cond_surface_point_LCS_norm_Z_cut),num = np.size(Femur_med_cond_surface_point_LCS_norm_Z_cut)), fp = Femur_med_cond_surface_point_LCS_norm_Z_cut)

                                    Femur_med_cond_surface_point_LCS_X_all = np.vstack((Femur_med_cond_surface_point_LCS_X_all,Femur_med_cond_surface_point_LCS_norm_X_cut_int))
                                    Femur_med_cond_surface_point_LCS_Y_all = np.vstack((Femur_med_cond_surface_point_LCS_Y_all,Femur_med_cond_surface_point_LCS_norm_Y_cut_int))
                                    Femur_med_cond_surface_point_LCS_Z_all = np.vstack((Femur_med_cond_surface_point_LCS_Z_all,Femur_med_cond_surface_point_LCS_norm_Z_cut_int))

                                    Tibia_lat_cond_surface_point_LCS_norm_X_cut_int = np.interp(x = np.linspace(0,np.size(Tibia_lat_cond_surface_point_LCS_norm_X_cut),num=100), xp = np.linspace(0,np.size(Tibia_lat_cond_surface_point_LCS_norm_X_cut),num = np.size(Tibia_lat_cond_surface_point_LCS_norm_X_cut)), fp = Tibia_lat_cond_surface_point_LCS_norm_X_cut)
                                    Tibia_lat_cond_surface_point_LCS_norm_Y_cut_int = np.interp(x = np.linspace(0,np.size(Tibia_lat_cond_surface_point_LCS_norm_Y_cut),num=100), xp = np.linspace(0,np.size(Tibia_lat_cond_surface_point_LCS_norm_Y_cut),num = np.size(Tibia_lat_cond_surface_point_LCS_norm_Y_cut)), fp = Tibia_lat_cond_surface_point_LCS_norm_Y_cut)
                                    Tibia_lat_cond_surface_point_LCS_norm_Z_cut_int = np.interp(x = np.linspace(0,np.size(Tibia_lat_cond_surface_point_LCS_norm_Z_cut),num=100), xp = np.linspace(0,np.size(Tibia_lat_cond_surface_point_LCS_norm_Z_cut),num = np.size(Tibia_lat_cond_surface_point_LCS_norm_Z_cut)), fp = Tibia_lat_cond_surface_point_LCS_norm_Z_cut)

                                    Tibia_lat_cond_surface_point_LCS_X_all = np.vstack((Tibia_lat_cond_surface_point_LCS_X_all,Tibia_lat_cond_surface_point_LCS_norm_X_cut_int))
                                    Tibia_lat_cond_surface_point_LCS_Y_all = np.vstack((Tibia_lat_cond_surface_point_LCS_Y_all,Tibia_lat_cond_surface_point_LCS_norm_Y_cut_int))
                                    Tibia_lat_cond_surface_point_LCS_Z_all = np.vstack((Tibia_lat_cond_surface_point_LCS_Z_all,Tibia_lat_cond_surface_point_LCS_norm_Z_cut_int))

                                    Tibia_med_cond_surface_point_LCS_norm_X_cut_int = np.interp(x = np.linspace(0,np.size(Tibia_med_cond_surface_point_LCS_norm_X_cut),num=100), xp = np.linspace(0,np.size(Tibia_med_cond_surface_point_LCS_norm_X_cut),num = np.size(Tibia_med_cond_surface_point_LCS_norm_X_cut)), fp = Tibia_med_cond_surface_point_LCS_norm_X_cut)
                                    Tibia_med_cond_surface_point_LCS_norm_Y_cut_int = np.interp(x = np.linspace(0,np.size(Tibia_med_cond_surface_point_LCS_norm_Y_cut),num=100), xp = np.linspace(0,np.size(Tibia_med_cond_surface_point_LCS_norm_Y_cut),num = np.size(Tibia_med_cond_surface_point_LCS_norm_Y_cut)), fp = Tibia_med_cond_surface_point_LCS_norm_Y_cut)
                                    Tibia_med_cond_surface_point_LCS_norm_Z_cut_int = np.interp(x = np.linspace(0,np.size(Tibia_med_cond_surface_point_LCS_norm_Z_cut),num=100), xp = np.linspace(0,np.size(Tibia_med_cond_surface_point_LCS_norm_Z_cut),num = np.size(Tibia_med_cond_surface_point_LCS_norm_Z_cut)), fp = Tibia_med_cond_surface_point_LCS_norm_Z_cut)

                                    Tibia_med_cond_surface_point_LCS_X_all = np.vstack((Tibia_med_cond_surface_point_LCS_X_all,Tibia_med_cond_surface_point_LCS_norm_X_cut_int))
                                    Tibia_med_cond_surface_point_LCS_Y_all = np.vstack((Tibia_med_cond_surface_point_LCS_Y_all,Tibia_med_cond_surface_point_LCS_norm_Y_cut_int))
                                    Tibia_med_cond_surface_point_LCS_Z_all = np.vstack((Tibia_med_cond_surface_point_LCS_Z_all,Tibia_med_cond_surface_point_LCS_norm_Z_cut_int))


                                    TTTG_d_norm_cut_int = np.interp(x = np.linspace(0,np.size(TTTG_d_norm_cut),num=100), xp = np.linspace(0,np.size(TTTG_d_norm_cut),num = np.size(TTTG_d_norm_cut)), fp = TTTG_d_norm_cut)
                                    TTTG_d_all = np.vstack((TTTG_d_all,TTTG_d_norm_cut_int))

                                    BO_perc_norm_cut_int = np.interp(x = np.linspace(0,np.size(BO_perc_norm_cut),num=100), xp = np.linspace(0,np.size(BO_perc_norm_cut),num = np.size(BO_perc_norm_cut)), fp = BO_perc_norm_cut)
                                    BO_perc_all = np.vstack((BO_perc_all,BO_perc_norm_cut_int))

                                    alpha_norm_cut_int = np.interp(x = np.linspace(0,np.size(alpha_norm_cut),num=100), xp = np.linspace(0,np.size(alpha_norm_cut),num = np.size(alpha_norm_cut)), fp = alpha_norm_cut)
                                    alpha_all = np.vstack((alpha_all,alpha_norm_cut_int))

                                    break
                            a = a +1        

                    print(np.size(X21_Tf_all,0), 'included')

                ### Eliminate the first row as has all zeros  ####

                X21_Tf_all = X21_Tf_all[1::,:]
                Y21_Tf_all = Y21_Tf_all[1::,:]
                Z21_Tf_all = Z21_Tf_all[1::,:]

                X21_Pf_all = X21_Pf_all[1::,:]
                Y21_Pf_all = Y21_Pf_all[1::,:]
                Z21_Pf_all = Z21_Pf_all[1::,:]


                Femur_surface_point_patella_LCS_X_all = Femur_surface_point_patella_LCS_X_all[1::,:]
                Femur_surface_point_patella_LCS_Y_all = Femur_surface_point_patella_LCS_Y_all[1::,:]
                Femur_surface_point_patella_LCS_Z_all = Femur_surface_point_patella_LCS_Z_all[1::,:]

                Patella_surface_point_LCS_X_all = Patella_surface_point_LCS_X_all[1::,:]
                Patella_surface_point_LCS_Y_all = Patella_surface_point_LCS_Y_all[1::,:]
                Patella_surface_point_LCS_Z_all = Patella_surface_point_LCS_Z_all[1::,:]

                Femur_lat_cond_surface_point_LCS_X_all = Femur_lat_cond_surface_point_LCS_X_all[1::,:]
                Femur_lat_cond_surface_point_LCS_Y_all = Femur_lat_cond_surface_point_LCS_Y_all[1::,:]
                Femur_lat_cond_surface_point_LCS_Z_all = Femur_lat_cond_surface_point_LCS_Z_all[1::,:]

                Femur_med_cond_surface_point_LCS_X_all = Femur_med_cond_surface_point_LCS_X_all[1::,:]
                Femur_med_cond_surface_point_LCS_Y_all = Femur_med_cond_surface_point_LCS_Y_all[1::,:]
                Femur_med_cond_surface_point_LCS_Z_all = Femur_med_cond_surface_point_LCS_Z_all[1::,:]

                Tibia_lat_cond_surface_point_LCS_X_all = Tibia_lat_cond_surface_point_LCS_X_all[1::,:]
                Tibia_lat_cond_surface_point_LCS_Y_all = Tibia_lat_cond_surface_point_LCS_Y_all[1::,:]
                Tibia_lat_cond_surface_point_LCS_Z_all = Tibia_lat_cond_surface_point_LCS_Z_all[1::,:]

                Tibia_med_cond_surface_point_LCS_X_all = Tibia_med_cond_surface_point_LCS_X_all[1::,:]
                Tibia_med_cond_surface_point_LCS_Y_all = Tibia_med_cond_surface_point_LCS_Y_all[1::,:]
                Tibia_med_cond_surface_point_LCS_Z_all = Tibia_med_cond_surface_point_LCS_Z_all[1::,:]


                TTTG_d_all = TTTG_d_all[1::,:]
                BO_perc_all = BO_perc_all[1::,:]
                alpha_all = alpha_all[1::,:]

                TTTG_d_all_diff = np.subtract(TTTG_d_all,(TTTG_d_all[:,0]).reshape(np.size(TTTG_d_all,0),1))
                BO_perc_all_diff = np.subtract(BO_perc_all,(BO_perc_all[:,0]).reshape(np.size(BO_perc_all,0),1))
                alpha_all_diff = np.subtract(alpha_all,(alpha_all[:,0]).reshape(np.size(alpha_all,0),1))

                Patella_translation_delta_X_all = Patella_surface_point_LCS_X_all
                Medial_cond_delta_X_all =  Tibia_med_cond_surface_point_LCS_X_all
                Lateral_cond_delta_X_all =  Tibia_lat_cond_surface_point_LCS_X_all

                Patella_translation_delta_Y_all = Patella_surface_point_LCS_Y_all
                Medial_cond_delta_Y_all = Tibia_med_cond_surface_point_LCS_Y_all
                Lateral_cond_delta_Y_all = Tibia_lat_cond_surface_point_LCS_Y_all

                Patella_translation_delta_Z_all = Patella_surface_point_LCS_Z_all
                Medial_cond_delta_Z_all = Tibia_med_cond_surface_point_LCS_Z_all
                Lateral_cond_delta_Z_all = Tibia_lat_cond_surface_point_LCS_Z_all

                TTTG_d_all_diff_mean = np.zeros(100);TTTG_d_all_diff_sd = np.zeros(100)
                BO_perc_all_diff_mean = np.zeros(100);BO_perc_all_diff_sd = np.zeros(100)
                alpha_all_diff_mean = np.zeros(100);alpha_all_diff_sd = np.zeros(100)


                for k in np.arange(0,100):
                    X21_Tf_all_mean[k] = np.mean(X21_Tf_all[:,k])
                    Y21_Tf_all_mean[k] = np.mean(Y21_Tf_all[:,k])
                    Z21_Tf_all_mean[k] = np.mean(Z21_Tf_all[:,k])

                    X21_Tf_all_sd[k] = np.std(X21_Tf_all[:,k])
                    Y21_Tf_all_sd[k] = np.std(Y21_Tf_all[:,k])
                    Z21_Tf_all_sd[k] = np.std(Z21_Tf_all[:,k])

                    X21_Pf_all_mean[k] = np.mean(X21_Pf_all[:,k])
                    Y21_Pf_all_mean[k] = np.mean(Y21_Pf_all[:,k])
                    Z21_Pf_all_mean[k] = np.mean(Z21_Pf_all[:,k])

                    X21_Pf_all_sd[k] = np.std(X21_Pf_all[:,k])
                    Y21_Pf_all_sd[k] = np.std(Y21_Pf_all[:,k])
                    Z21_Pf_all_sd[k] = np.std(Z21_Pf_all[:,k])

                    Patella_translation_delta_X_all_mean[k] = np.mean(Patella_translation_delta_X_all[:,k])
                    Medial_cond_delta_X_all_mean[k] = np.mean(Medial_cond_delta_X_all[:,k])
                    Lateral_cond_delta_X_all_mean[k] = np.mean(Lateral_cond_delta_X_all[:,k])

                    Patella_translation_delta_X_all_sd[k] = np.std(Patella_translation_delta_X_all[:,k])
                    Medial_cond_delta_X_all_sd[k] = np.std(Medial_cond_delta_X_all[:,k])
                    Lateral_cond_delta_X_all_sd[k] = np.std(Lateral_cond_delta_X_all[:,k])

                    Patella_translation_delta_Y_all_mean[k] = np.mean(Patella_translation_delta_Y_all[:,k])
                    Medial_cond_delta_Y_all_mean[k] = np.mean(Medial_cond_delta_Y_all[:,k])
                    Lateral_cond_delta_Y_all_mean[k] = np.mean(Lateral_cond_delta_Y_all[:,k])

                    Patella_translation_delta_Y_all_sd[k] = np.std(Patella_translation_delta_Y_all[:,k])
                    Medial_cond_delta_Y_all_sd[k] = np.std(Medial_cond_delta_Y_all[:,k])
                    Lateral_cond_delta_Y_all_sd[k] = np.std(Lateral_cond_delta_Y_all[:,k])

                    Patella_translation_delta_Z_all_mean[k] = np.mean(Patella_translation_delta_Z_all[:,k])
                    Medial_cond_delta_Z_all_mean[k] = np.mean(Medial_cond_delta_Z_all[:,k])
                    Lateral_cond_delta_Z_all_mean[k] = np.mean(Lateral_cond_delta_Z_all[:,k])

                    Patella_translation_delta_Z_all_sd[k] = np.std(Patella_translation_delta_Z_all[:,k])
                    Medial_cond_delta_Z_all_sd[k] = np.std(Medial_cond_delta_Z_all[:,k])
                    Lateral_cond_delta_Z_all_sd[k] = np.std(Lateral_cond_delta_Z_all[:,k])


                    TTTG_d_all_mean[k] = np.mean(TTTG_d_all[:,k])
                    TTTG_d_all_sd[k] = np.std(TTTG_d_all[:,k])

                    BO_perc_all_mean[k] = np.mean(BO_perc_all[:,k])
                    BO_perc_all_sd[k] = np.std(BO_perc_all[:,k])

                    alpha_all_mean[k] = np.mean(alpha_all[:,k])
                    alpha_all_sd[k] = np.std(alpha_all[:,k])

                    TTTG_d_all_diff_mean[k] = np.mean(TTTG_d_all_diff[:,k])
                    TTTG_d_all_diff_sd[k] = np.std(TTTG_d_all_diff[:,k])

                    BO_perc_all_diff_mean[k] = np.mean(BO_perc_all_diff[:,k])
                    BO_perc_all_diff_sd[k] = np.std(BO_perc_all_diff[:,k])

                    alpha_all_diff_mean[k] = np.mean(alpha_all_diff[:,k])
                    alpha_all_diff_sd[k] = np.std(alpha_all_diff[:,k])


                ci1X21_Tf= np.array(X21_Tf_all_mean + 1.96*X21_Tf_all_sd/np.sqrt(np.size(X21_Tf_all,0)))
                ci2X21_Tf= np.array(X21_Tf_all_mean - 1.96*X21_Tf_all_sd/np.sqrt(np.size(X21_Tf_all,0)))
                ci1Y21_Tf= np.array(Y21_Tf_all_mean + 1.96*Y21_Tf_all_sd/np.sqrt(np.size(X21_Tf_all,0)))
                ci2Y21_Tf= np.array(Y21_Tf_all_mean - 1.96*Y21_Tf_all_sd/np.sqrt(np.size(X21_Tf_all,0)))
                ci1Z21_Tf= np.array(Z21_Tf_all_mean + 1.96*Z21_Tf_all_sd/np.sqrt(np.size(X21_Tf_all,0)))
                ci2Z21_Tf= np.array(Z21_Tf_all_mean - 1.96*Z21_Tf_all_sd/np.sqrt(np.size(X21_Tf_all,0)))

                ci1X21_Pf= np.array(X21_Pf_all_mean + 1.96*X21_Pf_all_sd/np.sqrt(np.size(X21_Tf_all,0)))
                ci2X21_Pf= np.array(X21_Pf_all_mean - 1.96*X21_Pf_all_sd/np.sqrt(np.size(X21_Tf_all,0)))
                ci1Y21_Pf= np.array(Y21_Pf_all_mean + 1.96*Y21_Pf_all_sd/np.sqrt(np.size(X21_Tf_all,0)))
                ci2Y21_Pf= np.array(Y21_Pf_all_mean - 1.96*Y21_Pf_all_sd/np.sqrt(np.size(X21_Tf_all,0)))
                ci1Z21_Pf= np.array(Z21_Pf_all_mean + 1.96*Z21_Pf_all_sd/np.sqrt(np.size(X21_Tf_all,0)))
                ci2Z21_Pf= np.array(Z21_Pf_all_mean - 1.96*Z21_Pf_all_sd/np.sqrt(np.size(X21_Tf_all,0)))

                np.max(Y21_Tf_all_mean)
                np.max(ci1Y21_Tf)
                np.max(ci2Y21_Tf)


                ci1Patella_translation_delta_X= np.array(Patella_translation_delta_X_all_mean + 1.96*Patella_translation_delta_X_all_sd/np.sqrt(np.size(X21_Tf_all,0)))
                ci2Patella_translation_delta_X= np.array(Patella_translation_delta_X_all_mean - 1.96*Patella_translation_delta_X_all_sd/np.sqrt(np.size(X21_Tf_all,0)))
                ci1Medial_cond_delta_X= np.array(Medial_cond_delta_X_all_mean + 1.96*Medial_cond_delta_X_all_sd/np.sqrt(np.size(X21_Tf_all,0)))
                ci2Medial_cond_delta_X= np.array(Medial_cond_delta_X_all_mean - 1.96*Medial_cond_delta_X_all_sd/np.sqrt(np.size(X21_Tf_all,0)))
                ci1Lateral_cond_delta_X= np.array(Lateral_cond_delta_X_all_mean + 1.96*Lateral_cond_delta_X_all_sd/np.sqrt(np.size(X21_Tf_all,0)))
                ci2Lateral_cond_delta_X= np.array(Lateral_cond_delta_X_all_mean - 1.96*Lateral_cond_delta_X_all_sd/np.sqrt(np.size(X21_Tf_all,0)))

                ci1Patella_translation_delta_Y= np.array(Patella_translation_delta_Y_all_mean + 1.96*Patella_translation_delta_Y_all_sd/np.sqrt(np.size(X21_Tf_all,0)))
                ci2Patella_translation_delta_Y= np.array(Patella_translation_delta_Y_all_mean - 1.96*Patella_translation_delta_Y_all_sd/np.sqrt(np.size(X21_Tf_all,0)))
                ci1Medial_cond_delta_Y= np.array(Medial_cond_delta_Y_all_mean + 1.96*Medial_cond_delta_Y_all_sd/np.sqrt(np.size(X21_Tf_all,0)))
                ci2Medial_cond_delta_Y= np.array(Medial_cond_delta_Y_all_mean - 1.96*Medial_cond_delta_Y_all_sd/np.sqrt(np.size(X21_Tf_all,0)))
                ci1Lateral_cond_delta_Y= np.array(Lateral_cond_delta_Y_all_mean + 1.96*Lateral_cond_delta_Y_all_sd/np.sqrt(np.size(X21_Tf_all,0)))
                ci2Lateral_cond_delta_Y= np.array(Lateral_cond_delta_Y_all_mean - 1.96*Lateral_cond_delta_Y_all_sd/np.sqrt(np.size(X21_Tf_all,0)))

                ci1Patella_translation_delta_Z= np.array(Patella_translation_delta_Z_all_mean + 1.96*Patella_translation_delta_Z_all_sd/np.sqrt(np.size(X21_Tf_all,0)))
                ci2Patella_translation_delta_Z= np.array(Patella_translation_delta_Z_all_mean - 1.96*Patella_translation_delta_Z_all_sd/np.sqrt(np.size(X21_Tf_all,0)))
                ci1Medial_cond_delta_Z= np.array(Medial_cond_delta_Z_all_mean + 1.96*Medial_cond_delta_Z_all_sd/np.sqrt(np.size(X21_Tf_all,0)))
                ci2Medial_cond_delta_Z= np.array(Medial_cond_delta_Z_all_mean - 1.96*Medial_cond_delta_Z_all_sd/np.sqrt(np.size(X21_Tf_all,0)))
                ci1Lateral_cond_delta_Z= np.array(Lateral_cond_delta_Z_all_mean + 1.96*Lateral_cond_delta_Z_all_sd/np.sqrt(np.size(X21_Tf_all,0)))
                ci2Lateral_cond_delta_Z= np.array(Lateral_cond_delta_Z_all_mean - 1.96*Lateral_cond_delta_Z_all_sd/np.sqrt(np.size(X21_Tf_all,0)))


                ci1TTTG_d= np.array(TTTG_d_all_mean + 1.96*TTTG_d_all_sd/np.sqrt(np.size(X21_Tf_all,0)))
                ci2TTTG_d= np.array(TTTG_d_all_mean - 1.96*TTTG_d_all_sd/np.sqrt(np.size(X21_Tf_all,0)))

                ci1BO_perc= np.array(BO_perc_all_mean + 1.96*BO_perc_all_sd/np.sqrt(np.size(X21_Tf_all,0)))
                ci2BO_perc= np.array(BO_perc_all_mean - 1.96*BO_perc_all_sd/np.sqrt(np.size(X21_Tf_all,0)))

                ci1alpha= np.array(alpha_all_mean + 1.96*alpha_all_sd/np.sqrt(np.size(X21_Tf_all,0)))
                ci2alpha= np.array(alpha_all_mean - 1.96*alpha_all_sd/np.sqrt(np.size(X21_Tf_all,0)))

                ci1TTTG_d_diff= np.array(TTTG_d_all_diff_mean + 1.96*TTTG_d_all_diff_sd/np.sqrt(np.size(TTTG_d_all_diff,0)))
                ci2TTTG_d_diff= np.array(TTTG_d_all_diff_mean - 1.96*TTTG_d_all_diff_sd/np.sqrt(np.size(TTTG_d_all_diff,0)))

                ci1BO_perc_diff= np.array(BO_perc_all_diff_mean + 1.96*BO_perc_all_diff_sd/np.sqrt(np.size(BO_perc_all_diff,0)))
                ci2BO_perc_diff= np.array(BO_perc_all_diff_mean - 1.96*BO_perc_all_diff_sd/np.sqrt(np.size(BO_perc_all_diff,0)))

                ci1alpha_diff= np.array(alpha_all_diff_mean + 1.96*alpha_all_diff_sd/np.sqrt(np.size(alpha_all_diff,0)))
                ci2alpha_diff= np.array(alpha_all_diff_mean - 1.96*alpha_all_diff_sd/np.sqrt(np.size(alpha_all_diff,0)))


                
                if angle == -35:
                    angles_to_have = [-5,-10,-15,-20,-25,-30,-35]            
                if angle == -30:
                    angles_to_have = [-5,-10,-15,-20,-25,-30]
                elif angle == -25:
                    angles_to_have = [-5,-10,-15,-20,-25]
                elif angle == -20:
                    angles_to_have = [-5,-10,-15,-20]
                elif angle == -15:
                    angles_to_have = [-5,-10,-15]

                # angles_to_have = [-5,-10,-20,-30]
                inter_time_points = [0]

                for j in angles_to_have:
                    # print(j)
                    for i in np.arange(0,100):
                        if X21_Tf_all_mean[i]> j:
                            pass
                        else:
                            inter_time_points.append(int(i))
                            # print(i)
                            break


                inter_time_points.append(99)                    

                ### save time points (0,5,10,15,20,25,30) for each subject                      

                TTTG_d_all_int_tp = TTTG_d_all[:,inter_time_points]
                BO_perc_all_int_tp = BO_perc_all[:,inter_time_points]
                alpha_all_int_tp = alpha_all[:,inter_time_points]

                X21_Tf_all_int_tp = X21_Tf_all[:,inter_time_points]
                Y21_Tf_all_int_tp = Y21_Tf_all[:,inter_time_points]
                Z21_Tf_all_int_tp = Z21_Tf_all[:,inter_time_points]

                X21_Pf_all_int_tp = X21_Pf_all[:,inter_time_points]
                Y21_Pf_all_int_tp = Y21_Pf_all[:,inter_time_points]
                Z21_Pf_all_int_tp = Z21_Pf_all[:,inter_time_points]

                P_trans_X_all_int_tp = Patella_translation_delta_X_all[:,inter_time_points]
                P_trans_Y_all_int_tp = Patella_translation_delta_Y_all[:,inter_time_points]
                P_trans_Z_all_int_tp = Patella_translation_delta_Z_all[:,inter_time_points]
                
                
            


                if condition =='Knee':
                    fold_res = 'Results'

                elif condition == 'Knee_Patients':
                    fold_res = 'Results_PFPS'


                # path_results = f'{folder_with_results}/{fold_res}'
                path_results = '/Output/Results'
                # if not os.path.exists(path_results):
                #     os.makedirs(path_results)

                def extract_results_at_diff_angles(angle,variable):

                    if angle == -15:
                        result = variable[:,0:4]
                    elif angle == -20:
                        result = variable[:,4:5]
                    elif angle == -25:
                        result = variable[:,5:6]
                    elif angle == -30:
                        result = variable[:,6:7]
                    elif angle == -35:
                        result = variable[:,7:8]
                    return(result)



                reult_TTTG = extract_results_at_diff_angles(angle,TTTG_d_all_int_tp)
                output1 = path_results + '/' + 'TTTG_' + str(np.abs(angle)) + '.txt'
                np.savetxt(output1, reult_TTTG, delimiter=' ') 

                reult_BO = extract_results_at_diff_angles(angle,BO_perc_all_int_tp)
                output1 = path_results + 'BO_' + str(np.abs(angle)) + '.txt'
                np.savetxt(output1, reult_BO, delimiter=' ') 

                reult_LPT= extract_results_at_diff_angles(angle,alpha_all_int_tp)
                output1 = path_results  + 'LPT_' + str(np.abs(angle)) + '.txt'
                np.savetxt(output1, reult_LPT, delimiter=' ') 

                reult_Tf_X = extract_results_at_diff_angles(angle,X21_Tf_all_int_tp)
                output1 = path_results +  'Tf_X_' + str(np.abs(angle)) + '.txt'
                np.savetxt(output1, reult_Tf_X, delimiter=' ') 

                reult_Tf_Y = extract_results_at_diff_angles(angle,Y21_Tf_all_int_tp)
                output1 = path_results +  'Tf_Y_' + str(np.abs(angle)) + '.txt'
                np.savetxt(output1, reult_Tf_Y, delimiter=' ') 

                reult_Tf_Z = extract_results_at_diff_angles(angle,Z21_Tf_all_int_tp)
                output1 = path_results +  'Tf_Z_' + str(np.abs(angle)) + '.txt'
                np.savetxt(output1, reult_Tf_Z, delimiter=' ') 

                reult_Pf_X = extract_results_at_diff_angles(angle,X21_Pf_all_int_tp)
                output1 = path_results +  'Pf_X_' + str(np.abs(angle)) + '.txt'
                np.savetxt(output1, reult_Pf_X, delimiter=' ') 

                reult_Pf_Y = extract_results_at_diff_angles(angle,Y21_Pf_all_int_tp)
                output1 = path_results + 'Pf_Y_' + str(np.abs(angle)) + '.txt'
                np.savetxt(output1, reult_Pf_Y, delimiter=' ') 

                reult_Pf_Z = extract_results_at_diff_angles(angle,Z21_Pf_all_int_tp)
                output1 = path_results + 'Pf_Z_' + str(np.abs(angle)) + '.txt'
                np.savetxt(output1, reult_Pf_Z, delimiter=' ') 

                reult_P_trans_X = extract_results_at_diff_angles(angle,P_trans_X_all_int_tp)
                output1 = path_results +  'P_trans_X_' + str(np.abs(angle)) + '.txt'
                np.savetxt(output1, reult_P_trans_X, delimiter=' ') 

                reult_P_trans_Y = extract_results_at_diff_angles(angle,P_trans_Y_all_int_tp)
                output1 = path_results + 'P_trans_Y_' + str(np.abs(angle)) + '.txt'
                np.savetxt(output1, reult_P_trans_Y, delimiter=' ') 

                reult_P_trans_Z = extract_results_at_diff_angles(angle,P_trans_Z_all_int_tp)
                output1 = path_results + 'P_trans_Z_' + str(np.abs(angle)) + '.txt'
                np.savetxt(output1, reult_P_trans_Z, delimiter=' ') 

                #########  MORPHO CHAR  ##############


                # inter_time_points = [np.arange(0,100)]

                TTTG_d_all_mean_int_tp = TTTG_d_all_mean[inter_time_points]
                TTTG_d_all_sd_int_tp = TTTG_d_all_sd[inter_time_points]

                BO_perc_all_mean_int_tp = BO_perc_all_mean[inter_time_points]
                BO_perc_all_sd_int_tp = BO_perc_all_sd[inter_time_points]

                alpha_all_mean_int_tp = alpha_all_mean[inter_time_points]
                alpha_all_sd_int_tp = alpha_all_sd[inter_time_points]

                ci1TTTG_d_int_tp= ci1TTTG_d[inter_time_points]
                ci2TTTG_d_int_tp= ci2TTTG_d[inter_time_points]

                ci1BO_perc_int_tp= ci1BO_perc[inter_time_points]
                ci2BO_perc_int_tp= ci2BO_perc[inter_time_points]

                ci1alpha_int_tp= ci1alpha[inter_time_points]
                ci2alpha_int_tp= ci2alpha[inter_time_points]

                angles_to_have2 = np.reshape(angles_to_have,(np.size(angles_to_have),1))
                angles_to_have2 = np.vstack(([0],angles_to_have2))
                angles_to_have2 = np.reshape(angles_to_have2,(np.size(angles_to_have)+1,))
                

                Res_table = np.transpose(np.vstack((angles_to_have2,TTTG_d_all_mean_int_tp,TTTG_d_all_sd_int_tp,ci1TTTG_d_int_tp,ci2TTTG_d_int_tp,
                                        BO_perc_all_mean_int_tp,BO_perc_all_sd_int_tp,ci1BO_perc_int_tp,ci2BO_perc_int_tp,
                                        alpha_all_mean_int_tp,alpha_all_sd_int_tp,ci1alpha_int_tp,ci2alpha_int_tp)))
                
                
                

                import sys 

                stdoutOrigin=sys.stdout 
                sys.stdout = open(path_results + '/Results_' + Test + '_' + 'Morpho_' + str(np.abs(angle)) + '.txt', "w")
                print('Knee_Flexion','TTTG_Distance(mm)','Bisect_Offset(%)','Patellar_Tilt(°)')
                for i in np.arange(0,np.size(inter_time_points)):

                    ##two decimal
                    # print(angles_to_have2[i], str('{0:.2f}'.format(TTTG_d_all_mean_int_tp[i])) + '[' + str('{0:.2f}'.format(ci2TTTG_d_int_tp[i])) + '-' + str('{0:.2f}'.format(ci1TTTG_d_int_tp[i])) + ']', str('{0:.2f}'.format(BO_perc_all_mean_int_tp[i])) + '[' + str('{0:.2f}'.format(ci2BO_perc_int_tp[i])) + '-' + str('{0:.2f}'.format(ci1BO_perc_int_tp[i])) + ']',
                    # str('{0:.2f}'.format(alpha_all_mean_int_tp[i])) + '[' + str('{0:.2f}'.format(ci2alpha_int_tp[i])) + '-' + str('{0:.2f}'.format(ci1alpha_int_tp[i])) + ']')

                    ##one decimal
                    print(angles_to_have2[i], str('{0:.1f}'.format(TTTG_d_all_mean_int_tp[i])) + '[' + str('{0:.1f}'.format(ci2TTTG_d_int_tp[i])) + '-' + str('{0:.1f}'.format(ci1TTTG_d_int_tp[i])) + ']', str('{0:.1f}'.format(BO_perc_all_mean_int_tp[i])) + '[' + str('{0:.1f}'.format(ci2BO_perc_int_tp[i])) + '-' + str('{0:.1f}'.format(ci1BO_perc_int_tp[i])) + ']',
                    str('{0:.1f}'.format(alpha_all_mean_int_tp[i])) + '[' + str('{0:.1f}'.format(ci2alpha_int_tp[i])) + '-' + str('{0:.1f}'.format(ci1alpha_int_tp[i])) + ']')

                sys.stdout.close()
                sys.stdout=stdoutOrigin



                import pandas as pd

                df = pd.DataFrame(Res_table)

                # Create a Pandas Excel writer using XlsxWriter as the engine.
                writer = pd.ExcelWriter(folder_with_results + '/Results/Ortho_metrics_mean_sd_ci'+ str(np.abs(angle)) + '.xlsx')

                column_name = ['Angles','TTTG_mean','TTTG_SD','TTTG_CI_UP','TTTG_CI_LO','BO_mean','BO_SD','BO_CI_UP','BO_CI_LO','LPT_mean','LPT_SD','LPT_CI_UP','LPT_CI_LO']
                # Convert the dataframe to an XlsxWriter Excel object.
                df.to_excel(writer, sheet_name='Sheet1',header=column_name)

                # Close the Pandas Excel writer and output the Excel file.
                # writer.save() 
                writer.close()


                ## difference compare to 0 deg flexion

                TTTG_d_all_diff_mean_int_tp = TTTG_d_all_diff_mean[inter_time_points]
                TTTG_d_all_diff_sd_int_tp = TTTG_d_all_diff_sd[inter_time_points]

                BO_perc_all_diff_mean_int_tp = BO_perc_all_diff_mean[inter_time_points]
                BO_perc_all_diff_sd_int_tp = BO_perc_all_diff_sd[inter_time_points]

                alpha_all_diff_mean_int_tp = alpha_all_diff_mean[inter_time_points]
                alpha_all_diff_sd_int_tp = alpha_all_diff_sd[inter_time_points]


                ci1TTTG_d_diff_int_tp= ci1TTTG_d_diff[inter_time_points]
                ci2TTTG_d_diff_int_tp= ci2TTTG_d_diff[inter_time_points]

                ci1BO_perc_diff_int_tp= ci1BO_perc_diff[inter_time_points]
                ci2BO_perc_diff_int_tp= ci2BO_perc_diff[inter_time_points]

                ci1alpha_diff_int_tp= ci1alpha_diff[inter_time_points]
                ci2alpha_diff_int_tp= ci2alpha_diff[inter_time_points]

                Res_table_diff = np.transpose(np.vstack((TTTG_d_all_diff_mean_int_tp,TTTG_d_all_diff_sd_int_tp,ci1TTTG_d_diff_int_tp,
                                        ci2TTTG_d_diff_int_tp,BO_perc_all_diff_mean_int_tp,BO_perc_all_diff_sd_int_tp,
                                        ci1BO_perc_diff_int_tp,ci2BO_perc_diff_int_tp,alpha_all_diff_mean_int_tp,alpha_all_diff_sd_int_tp,
                                        ci1alpha_diff_int_tp,ci2alpha_diff_int_tp)))

                stdoutOrigin=sys.stdout 
                sys.stdout = open(path_results + '/Results_' + Test + '_' + 'Morpho_difference_to_0_deg_' + str(np.abs(angle)) + '.txt', "w")
                print('Knee_Flexion','TTTG_Distance(mm)DIFF','Bisect_Offset(%)DIFF','Patellar_Tilt(°)DIFF')
                for i in np.arange(0,np.size(inter_time_points)):

                    ##two decimal
                    # print(angles_to_have2[i], str('{0:.2f}'.format(TTTG_d_all_diff_mean_int_tp[i])) + '[' + str('{0:.2f}'.format(ci2TTTG_d_int_tp[i])) + '-' + str('{0:.2f}'.format(ci1TTTG_d_int_tp[i])) + ']', str('{0:.2f}'.format(BO_perc_all_diff_mean_int_tp[i])) + '[' + str('{0:.2f}'.format(ci2BO_perc_int_tp[i])) + '-' + str('{0:.2f}'.format(ci1BO_perc_int_tp[i])) + ']',
                    # str('{0:.2f}'.format(alpha_all_diff_mean_int_tp[i])) + '[' + str('{0:.2f}'.format(ci2alpha_int_tp[i])) + '-' + str('{0:.2f}'.format(ci1alpha_int_tp[i])) + ']')

                    ##one decimal
                    print(angles_to_have2[i], str('{0:.1f}'.format(TTTG_d_all_diff_mean_int_tp[i])) + '[' + str('{0:.1f}'.format(ci2TTTG_d_diff_int_tp[i])) + '-' + str('{0:.1f}'.format(ci1TTTG_d_diff_int_tp[i])) + ']', str('{0:.1f}'.format(BO_perc_all_diff_mean_int_tp[i])) + '[' + str('{0:.1f}'.format(ci2BO_perc_diff_int_tp[i])) + '-' + str('{0:.1f}'.format(ci1BO_perc_diff_int_tp[i])) + ']',
                    str('{0:.1f}'.format(alpha_all_diff_mean_int_tp[i])) + '[' + str('{0:.1f}'.format(ci2alpha_diff_int_tp[i])) + '-' + str('{0:.1f}'.format(ci1alpha_diff_int_tp[i])) + ']')

                sys.stdout.close()
                sys.stdout=stdoutOrigin


                if np.size(angle_maxs) > 1:
                    print('loop angle')
                    x1 = np.linspace(0,99,100)

                    x = np.linspace(0,-angle,100)
                    
#                     save interpolated mean cardan angles tibifemoral
                    
                    cardan_angles_Tf[:,0]=X21_Tf_all_mean
                    cardan_angles_Tf[:,1]=Y21_Tf_all_mean
                    cardan_angles_Tf[:,2]=Z21_Tf_all_mean

                    cardan_angles_Pf[:,0]=X21_Pf_all_mean
                    cardan_angles_Pf[:,1]=Y21_Pf_all_mean
                    cardan_angles_Pf[:,2]=Z21_Pf_all_mean

                    txtfilename_Tf=f'{path_results}/Mean_cardan_angles_Tf_{angle}.txt'
                    txtfilename_Pf=f'{path_results}/Mean_cardan_angles_Pf_{angle}.txt'

                    csvfilename_Tf=f'{path_results}/Mean_cardan_angles_Tf_{angle}.csv'
                    csvfilename_Pf=f'{path_results}/Mean_cardan_angles_Pf_{angle}.csv'
                    
                    
                    interp=np.arange(1,100)
                    save_Cardan_angles(cardan_angles_Tf, txtfilename_Tf,csvfilename_Tf,interp)
                    save_Cardan_angles(cardan_angles_Pf, txtfilename_Pf,csvfilename_Pf,interp)

                else:     

                    x1 = np.linspace(0,99,100)

                    x = np.linspace(0,-angle,100)
                    
#                     save interpolated mean cardan angles tibifemoral
                    
                    cardan_angles_Tf[:,0]=X21_Tf_all_mean
                    cardan_angles_Tf[:,1]=Y21_Tf_all_mean
                    cardan_angles_Tf[:,2]=Z21_Tf_all_mean
                
                    cardan_angles_Pf[:,0]=X21_Pf_all_mean
                    cardan_angles_Pf[:,1]=Y21_Pf_all_mean
                    cardan_angles_Pf[:,2]=Z21_Pf_all_mean
                
                    
                    txtfilename_Tf=f'{path_results}/Mean_cardan_angles_Tf.txt'
                    txtfilename_Pf=f'{path_results}/Mean_cardan_angles_Pf.txt'
                    
                    csvfilename_Tf=f'{path_results}/Mean_cardan_angles_Tf.csv'
                    csvfilename_Pf=f'{path_results}/Mean_cardan_angles_Pf.csv'
                    
                    
                    interp=np.arange(1,100)
                    save_Cardan_angles(cardan_angles_Tf, txtfilename_Tf,csvfilename_Tf,interp)
                    save_Cardan_angles(cardan_angles_Pf, txtfilename_Pf,csvfilename_Pf,interp)
                    

                    figure = plt.figure()
                    plt.suptitle('TIBIOFEMORAL JOINT - From full extension to ' + str(np.abs(angle)) + '˚ of flexion')
                    plt.subplots_adjust(hspace =0.25, wspace = 0.25, top=0.914)
                    #L,U = -10,10

                    O=0
                    plt.subplot(231)    
                    plt.plot (x1,np.transpose(X21_Tf_all_mean), color = 'r')
                    plt.plot (x1,np.transpose(ci1X21_Tf), color = 'r', alpha =0.3, linewidth=0.5)
                    plt.plot (x1,np.transpose(ci2X21_Tf), color = 'r', alpha =0.3, linewidth=0.5)
                    plt.fill_between(x1,ci1X21_Tf,ci2X21_Tf, alpha=0.1, color = 'r')
                    plt.title('F/E-axis'), plt.ylabel('Rotation angles [95% CI] (°)'),plt.xlabel('Motion(%)')
                    #
                    plt.subplot(232)    
                    plt.plot (np.transpose(Y21_Tf_all_mean), color = 'g')
                    plt.plot (np.transpose(ci1Y21_Tf), color = 'g', alpha =0.3, linewidth=0.5)
                    plt.plot (np.transpose(ci2Y21_Tf), color = 'g', alpha =0.3, linewidth=0.5)
                    plt.fill_between(x1,ci1Y21_Tf,ci2Y21_Tf, alpha=0.1, color = 'g')
                    plt.title('ADD/ABD-axis'), plt.ylabel('Rotation angles  [95% CI] (°)'),plt.xlabel('Knee Flexion(˚)')
                    plt.ylim((-2,5))

                    plt.subplot(233)    
                    plt.plot (np.transpose(Z21_Tf_all_mean), color = 'b')
                    plt.plot (np.transpose(ci1Z21_Tf), color = 'b', alpha =0.3, linewidth=0.5)
                    plt.plot (np.transpose(ci2Z21_Tf), color = 'b', alpha =0.3, linewidth=0.5)
                    plt.fill_between(x1,ci1Z21_Tf,ci2Z21_Tf, alpha=0.1, color = 'b')
                    plt.title('INT/EXT-axis'), plt.ylabel('Rotation angles  [95% CI] (°)'),plt.xlabel('Knee Flexion(˚)')
                    #plt.ylim((0,15))

                    plt.subplot(234)    
                    plt.plot (np.transpose(X21_Pf_all_mean), color = 'r')
                    plt.plot (np.transpose(ci1X21_Pf), color = 'r', alpha =0.3, linewidth=0.5)
                    plt.plot (np.transpose(ci2X21_Pf), color = 'r', alpha =0.3, linewidth=0.5)
                    plt.fill_between(x1,ci1X21_Pf,ci2X21_Pf, alpha=0.1, color = 'r')
                    plt.title('X-axis'), plt.ylabel('Angles [95% CI] (°)')
                    #
                    plt.subplot(235)    
                    plt.plot (np.transpose(Y21_Pf_all_mean), color = 'g')
                    plt.plot (np.transpose(ci1Y21_Pf), color = 'g', alpha =0.3, linewidth=0.5)
                    plt.plot (np.transpose(ci2Y21_Pf), color = 'g', alpha =0.3, linewidth=0.5)
                    plt.fill_between(x1,ci1Y21_Pf,ci2Y21_Pf, alpha=0.1, color = 'g')
                    plt.title('Y-axis'), plt.ylabel('Angles [95% CI] (°)'),plt.xlabel('Motion(%)')
                    # plt.ylim((-10,5))

                    plt.subplot(236)    
                    plt.plot (np.transpose(Z21_Pf_all_mean), color = 'b')
                    plt.plot (np.transpose(ci1Z21_Pf), color = 'b', alpha =0.3, linewidth=0.5)
                    plt.plot (np.transpose(ci2Z21_Pf), color = 'b', alpha =0.3, linewidth=0.5)
                    plt.fill_between(x1,ci1Z21_Pf,ci2Z21_Pf, alpha=0.1, color = 'b')
                    plt.title('Z-axis'), plt.ylabel('Angles [95% CI] (°)')


                            ###  PLOT ANGLE VS FLEXION


                    figure = plt.figure()

                    if point == 'points':
                        color1 = 'black'
                        color2 = 'black'
                        color3 = 'black'

                        color1 = 'b'
                        color2 = 'b'
                        color3 = 'b'

                        label = 'Manual landmark approach'
                        linestyle='--'

                    elif point == 'points_mean':

                        color1 = 'r'
                        color2 = 'g'
                        color3 = 'b'

                        color1 = 'r'
                        color2 = 'r'
                        color3 = 'r'


                        label = 'Proposed approach'
                        linestyle='-'


                    plt.subplot(231)    
                    plt.plot (x1,np.transpose(X21_Tf_all_mean), color = color1,linestyle=linestyle)
                    plt.plot (x1,np.transpose(ci1X21_Tf), color = color1, alpha =0.3, linewidth=0.5)
                    plt.plot (x1,np.transpose(ci2X21_Tf), color = color1, alpha =0.3, linewidth=0.5)
                    plt.fill_between(x1,ci1X21_Tf,ci2X21_Tf, alpha=0.1, color = color1)
                    plt.title('F/E-axis'), plt.ylabel('Rotation angles [95% CI] (°)'),plt.xlabel('Motion(%)')
                    #

                    plt.subplot(232)    
                    plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(Y21_Tf_all_mean), color = color2,linestyle=linestyle)
                    plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(ci1Y21_Tf), color = color2, alpha =0.3, linewidth=0.5,linestyle=linestyle)
                    plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(ci2Y21_Tf), color = color2, alpha =0.3, linewidth=0.5,linestyle=linestyle)
                    plt.fill_between(-np.transpose(X21_Tf_all_mean),ci1Y21_Tf,ci2Y21_Tf, alpha=0.1, color = color2)
                    plt.title('ADD/ABD-axis'), plt.ylabel('Rotation angles  [95% CI] (°)'),plt.xlabel('Knee Flexion(˚)')
                    plt.ylim((-5,5))


                    plt.subplot(233)    
                    plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(Z21_Tf_all_mean), color = color3,linestyle=linestyle)
                    plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(ci1Z21_Tf), color = color3, alpha =0.3, linewidth=0.5,linestyle=linestyle)
                    plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(ci2Z21_Tf), color = color3, alpha =0.3, linewidth=0.5,linestyle=linestyle)
                    plt.fill_between(-np.transpose(X21_Tf_all_mean),ci1Z21_Tf,ci2Z21_Tf, alpha=0.1, color = color3)
                    plt.title('INT/EXT-axis'), plt.ylabel('Rotation angles  [95% CI] (°)'),plt.xlabel('Knee Flexion(˚)')
                    plt.ylim((-0.3,10))

                    plt.legend()


                    # plt.legend(label,loc='upper right', bbox_to_anchor=(1.3, 1.3))


                    # figure = plt.figure()
                    plt.subplots_adjust(hspace =0.25, wspace = 0.25)
                    #L,U = -10,10

                    O=0
                    plt.subplot(234)    
                    plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(X21_Pf_all_mean), color = 'r')
                    plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(ci1X21_Pf), color = 'r', alpha =0.3, linewidth=0.5)
                    plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(ci2X21_Pf), color = 'r', alpha =0.3, linewidth=0.5)
                    plt.fill_between(-np.transpose(X21_Tf_all_mean),ci1X21_Pf,ci2X21_Pf, alpha=0.1, color = 'r')
                    plt.title('X-axis'), plt.ylabel('Angles [95% CI] (°)'),plt.xlabel('Knee Flexion(˚)')
                    #
                    plt.subplot(235)    
                    plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(Y21_Pf_all_mean), color = 'g')
                    plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(ci1Y21_Pf), color = 'g', alpha =0.3, linewidth=0.5)
                    plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(ci2Y21_Pf), color = 'g', alpha =0.3, linewidth=0.5)
                    plt.fill_between(-np.transpose(X21_Tf_all_mean),ci1Y21_Pf,ci2Y21_Pf, alpha=0.1, color = 'g')
                    plt.title('Y-axis'), plt.ylabel('Angles [95% CI] (°)'),plt.xlabel('Knee Flexion(˚)')
                    # plt.ylim((-10,5))

                    plt.subplot(236)    
                    plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(Z21_Pf_all_mean), color = 'b')
                    plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(ci1Z21_Pf), color = 'b', alpha =0.3, linewidth=0.5)
                    plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(ci2Z21_Pf), color = 'b', alpha =0.3, linewidth=0.5)
                    plt.fill_between(-np.transpose(X21_Tf_all_mean),ci1Z21_Pf,ci2Z21_Pf, alpha=0.1, color = 'b')
                    plt.title('Z-axis'), plt.ylabel('Angles [95% CI] (°)'),plt.xlabel('Knee Flexion(˚)')


                #     #### PLOT individual subjects  ####

                    figure = plt.figure()
                    plt.suptitle('KNEE JOINT - From full extension to ' + str(np.abs(angle)) + ' degrees of flexion')
                    plt.subplots_adjust(hspace =0.4, wspace = 0.25)

                    plt.subplot(231); plt.plot( np.transpose(X21_Tf_all))
                    #plt.xlabel('task (%)')
                    plt.ylabel('angles (deg)')
                    plt.title('TIBIO-FEMORAL JOINT X-axis')
                    plt.subplot(232); plt.plot(np.transpose(Y21_Tf_all))
                    #plt.xlabel('task (%)')
                    #plt.ylabel('angles (deg)')
                    plt.title('TIBIO-FEMORAL JOINT Y-axis')
                    plt.subplot(233); plt.plot(np.transpose(Z21_Tf_all))
                    #plt.xlabel('task (%)')
                    #plt.ylabel('angles (deg)')
                    plt.title('TIBIO-FEMORAL JOINT Z-axis')

                    plt.legend(subjects_included,loc='upper right', bbox_to_anchor=(1.3, 1.3))

                    plt.subplot(234); plt.plot(np.transpose(X21_Pf_all))
                    #plt.xlabel('task (%)')
                    plt.ylabel('angles (deg)')
                    plt.title('TIBIO-FEMORAL JOINT X-axis')
                    plt.subplot(235); plt.plot(np.transpose(Y21_Pf_all))
                    #plt.xlabel('task (%)')
                    #plt.ylabel('angles (deg)')
                    plt.title('TIBIO-FEMORAL JOINT Y-axis')
                    plt.subplot(236); plt.plot(np.transpose(Z21_Pf_all))
                    #plt.xlabel('task (%)')
                    #plt.ylabel('angles (deg)')
                    plt.title('TIBIO-FEMORAL JOINT Z-axis')

                    # # plt.legend(subjects_included)



                    #####################################################
                    ####### MORPHOLOGICAL CHARACTERISTICS   #########
                    #####################################################

                    x = np.linspace(0,99,100)

                    figure = plt.figure()
                    plt.suptitle('MORPHOLOGICAL CHARACTERISTICS')
                    plt.subplots_adjust(hspace =0.25, wspace = 0.25)
                    #L,U = -10,10


                    color = 'r'
                    linestyle='-'
                    label='Concentric'
                    label='Eccentric'

                    O=0
                    plt.subplot(231)    
                    plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(TTTG_d_all_mean), color = color,linestyle=linestyle)
                    plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(ci1TTTG_d), color = color, alpha =0.3, linewidth=0.5)
                    plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(ci2TTTG_d), color = color, alpha =0.3, linewidth=0.5)
                    plt.fill_between(-np.transpose(X21_Tf_all_mean),ci1TTTG_d,ci2TTTG_d, alpha=0.1, color = color)
                    plt.title('TTTG', size=14), plt.ylabel('Distance [95% CI] (mm)', size=12),plt.xlabel('Knee Flexion(˚)', size=14)
                    plt.tick_params(axis='both', labelsize=12)


                    plt.subplot(232)    
                    plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(BO_perc_all_mean), color = color,linestyle=linestyle)
                    plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(ci1BO_perc), color = color, alpha =0.3, linewidth=0.5)
                    plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(ci2BO_perc), color = color, alpha =0.3, linewidth=0.5)
                    plt.fill_between(-np.transpose(X21_Tf_all_mean),ci1BO_perc,ci2BO_perc, alpha=0.1, color = color)
                    plt.title('BO', size=14), plt.ylabel('Percentage BO [95% CI] (%)', size=12),plt.xlabel('Knee Flexion(˚)', size=14)
                    plt.tick_params(axis='both', labelsize=12)

                    plt.subplot(233)    
                    line1 = plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(alpha_all_mean), color = color, linestyle=linestyle, label=label)
                    plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(ci1alpha), color = color, alpha =0.3, linewidth=0.5)
                    plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(ci2alpha), color = color, alpha =0.3, linewidth=0.5)
                    plt.fill_between(-np.transpose(X21_Tf_all_mean),ci1alpha,ci2alpha, alpha=0.1, color = color)
                    plt.title('LATERAL PATELLAR TILT', size=14), plt.ylabel('Angles [95% CI] (°)', size=12),plt.xlabel('Knee Flexion(˚)', size=14)
                    # plt.legend()
                    plt.tick_params(axis='both', labelsize=12)


        #              Lateral_trochlear_inclination
        #         Sulcus_depth_d  
        # sulcus_angle
                    # #### Individual curves


                    figure = plt.figure()
                    plt.suptitle('MORPHOLOGICAL CHARACTERISTICS')
                    plt.subplots_adjust(hspace =0.25, wspace = 0.25)

                    plt.subplot(231)    
                    plt.plot (np.transpose(TTTG_d_all))
                    plt.title('TTTG'), plt.ylabel('Distance [95% CI] (mm)')
                    #
                    plt.subplot(232)    
                    plt.plot (np.transpose(BO_perc_all))
                    plt.title('BO'), plt.ylabel('Percentage BO [95% CI] (%)'),plt.xlabel('Motion(%)')
                    # plt.ylim((-5,5))

                    plt.subplot(233)    
                    plt.plot (np.transpose(alpha_all))
                    plt.title('ALPHA ANGLE'), plt.ylabel('Angles [95% CI] (°)')
                    #plt.ylim((0,15))

                    plt.legend(subjects_included,loc='upper right', bbox_to_anchor=(1.3, 1.3))


              ## BUZZATTI  

                    x = np.linspace(0,99,100)

                    figure = plt.figure()
                    plt.suptitle('MORPHOLOGICAL CHARACTERISTICS', size=20)
                    plt.subplots_adjust(hspace =0.25, wspace = 0.25)
                    #L,U = -10,10


                    O=0
                    plt.subplot(231)    
                    plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(TTTG_d_all_mean), color = 'r')
                    # plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(ci1TTTG_d), color = 'r', alpha =0.3, linewidth=0.5)
                    # plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(ci2TTTG_d), color = 'r', alpha =0.3, linewidth=0.5)
                    # plt.fill_between(-np.transpose(X21_Tf_all_mean),ci1TTTG_d,ci2TTTG_d, alpha=0.1, color = 'r')
                    plt.title('TTTG'), plt.ylabel('Distance [95% CI] (mm)'),plt.xlabel('Knee Flexion(˚)')
                    #
                    plt.subplot(232)    
                    plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(BO_perc_all_mean), color = 'r')
                    # plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(ci1BO_perc), color = 'r', alpha =0.3, linewidth=0.5)
                    # plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(ci2BO_perc), color = 'r', alpha =0.3, linewidth=0.5)
                    # plt.fill_between(-np.transpose(X21_Tf_all_mean),ci1BO_perc,ci2BO_perc, alpha=0.1, color = 'r')
                    plt.title('BO'), plt.ylabel('Percentage BO [95% CI] (%)'),plt.xlabel('Knee Flexion(˚)')
                    # plt.ylim((-5,5))

                    plt.subplot(233)    
                    line1 = plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(alpha_all_mean), color = 'r')
                    # plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(ci1alpha), color = 'r', alpha =0.3, linewidth=0.5)
                    # plt.plot (-np.transpose(X21_Tf_all_mean),np.transpose(ci2alpha), color = 'r', alpha =0.3, linewidth=0.5)
                    # plt.fill_between(-np.transpose(X21_Tf_all_mean),ci1alpha,ci2alpha, alpha=0.1, color = 'r')
                    plt.title('LATERAL PATELLAR TILT'), plt.ylabel('Angles [95% CI] (°)'),plt.xlabel('Knee Flexion(˚)')
                    plt.legend()
                    #plt.ylim((0,15))
