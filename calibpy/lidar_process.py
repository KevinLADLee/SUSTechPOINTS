
import numpy as np
import open3d
import json
import math


# def euler_angle_to_rotate_matrix(eu, t):
#     theta = eu
#     #Calculate rotation about x axis
#     R_x = np.array([
#         [1,       0,              0],
#         [0,       math.cos(theta[0]),   -math.sin(theta[0])],
#         [0,       math.sin(theta[0]),   math.cos(theta[0])]
#     ])

#     #Calculate rotation about y axis
#     R_y = np.array([
#         [math.cos(theta[1]),      0,      math.sin(theta[1])],
#         [0,                       1,      0],
#         [-math.sin(theta[1]),     0,      math.cos(theta[1])]
#     ])

#     #Calculate rotation about z axis
#     R_z = np.array([
#         [math.cos(theta[2]),    -math.sin(theta[2]),      0],
#         [math.sin(theta[2]),    math.cos(theta[2]),       0],
#         [0,               0,                  1]])

#     R = np.matmul(R_x, np.matmul(R_y, R_z))

#     t = t.reshape([-1,1])
#     R = np.concatenate([R,t], axis=-1)
#     R = np.concatenate([R, np.array([0,0,0,1]).reshape([1,-1])], axis=0)
#     return R


#  euler_angle_to_rotate_matrix(np.array([0, np.pi/3, np.pi/2]), np.array([1,2,3]))
# def psr_to_xyz(p,s,r):
#     trans_matrix = euler_angle_to_rotate_matrix(r, p)

#     x=s[0]/2
#     y=s[1]/2
#     z=s[2]/2
    

#     local_coord = np.array([
#         x, y, -z, 1,   x, -y, -z, 1,  #front-left-bottom, front-right-bottom
#         x, -y, z, 1,   x, y, z, 1,    #front-right-top,   front-left-top

#         -x, y, -z, 1,   -x, -y, -z, 1,#rear-left-bottom, rear-right-bottom
#         -x, -y, z, 1,   -x, y, z, 1,  #rear-right-top,   rear-left-top
        
#         #middle plane
#         #0, y, -z, 1,   0, -y, -z, 1,  #rear-left-bottom, rear-right-bottom
#         #0, -y, z, 1,   0, y, z, 1,    #rear-right-top,   rear-left-top
#         ]).reshape((-1,4))

#     world_coord = np.matmul(trans_matrix, np.transpose(local_coord))
    
#     return world_coord            


def euler_angle_to_rotate_matrix(eu):
    theta = eu
    #Calculate rotation about x axis
    R_x = np.array([
        [1,       0,              0],
        [0,       math.cos(theta[0]),   -math.sin(theta[0])],
        [0,       math.sin(theta[0]),   math.cos(theta[0])]
    ])

    #Calculate rotation about y axis
    R_y = np.array([
        [math.cos(theta[1]),      0,      math.sin(theta[1])],
        [0,                       1,      0],
        [-math.sin(theta[1]),     0,      math.cos(theta[1])]
    ])

    #Calculate rotation about z axis
    R_z = np.array([
        [math.cos(theta[2]),    -math.sin(theta[2]),      0],
        [math.sin(theta[2]),    math.cos(theta[2]),       0],
        [0,               0,                  1]])

    R = np.matmul(R_x, np.matmul(R_y, R_z))
    return R


def box_to_nparray(box):
    return np.array([
        [box["position"]["x"], box["position"]["y"], box["position"]["z"]],
        [box["scale"]["x"], box["scale"]["y"], box["scale"]["z"]],
        [box["rotation"]["x"], box["rotation"]["y"], box["rotation"]["z"]],
    ])

def rot_mat(vecA,vecB = [0,0,1]):
# a and b are in the form of numpy array

    vecB = np.asarray(vecB).reshape(3,1)
    vecA = np.asarray(vecA).reshape(3,1)
    norm_va = np.linalg.norm(vecA)
    va = vecA - norm_va * vecB
    u = va / np.linalg.norm(va)
    H = np.eye(3) - 2 * u @ u.T
    H2 = np.linalg.inv(H)
    return H2

WORKDIR = '/home/kevinlad/project/points-calibration/SUSTechPOINTS/data/calib-test'

print (WORKDIR + '/lidar/1603344898.141445160.pcd')

raw_cloud = open3d.io.read_point_cloud(WORKDIR + "/lidar/1603344898.141445160.pcd")

print (raw_cloud)

label = None
with open(WORKDIR + "/label/1603344898.141445160.json") as f:
  label = json.load(f)

print(type(label))
print(type(label[1]))

print("-----")

crop_cloud = None
for l in label:
    if l['obj_type'] == 'CalibBoard':
        np_box = box_to_nparray(l['psr'])
        # print(psr_to_xyz(np_box[0], np_box[1] ,np_box[2]))
        print("-----")
        box = open3d.geometry.OrientedBoundingBox(np_box[0],euler_angle_to_rotate_matrix(np_box[2]),np_box[1])
        crop_cloud = raw_cloud.crop(box)

crop_cloud.paint_uniform_color([1, 0.706, 0])
# open3d.visualization.draw_geometries([crop_cloud])

plane_model, inliers = crop_cloud.segment_plane(0.02, 3, 10) 

inlier_cloud = crop_cloud.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud = crop_cloud.select_by_index(inliers, invert=True)
outlier_cloud.paint_uniform_color([0, 1.0, 0])
print(outlier_cloud)

# 

print(plane_model)


rotation = rot_mat(plane_model[0:3])

arrow = open3d.geometry.TriangleMesh.create_arrow(0.1, 0.15, 0.5, 0.4)
arrow.rotate(rotation, [0,0,0])
arrow.translate(crop_cloud.get_center())

open3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, arrow])

