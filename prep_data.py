import numpy as np
import open3d as o3d
import os
import pyquaternion
#import matplotlib.pyplot as plt
import copy as copy
def read_camera_poses(filename):
    with open(filename) as file:
        poses=[]
        for line in file:
            elems = line.rstrip().split(' ')
            mat = []
            for p in elems:
                if p == '':
                    continue
                mat.append(float(p))

            #print(mat[1:4])
            #print(mat[4:])
            position = np.asarray(mat[1:4])
            rotation = np.asarray(mat[4:])

            M = np.eye(4)
            M[0, 0] = -1.
            M[1, 1] = 1.
            M[2, 2] = 1.

            qw = rotation[3]
            qx = rotation[0]
            qy = rotation[1]
            qz = rotation[2]

            quaternion = pyquaternion.Quaternion(qw, qx, qy, qz)
            rotation = quaternion.rotation_matrix

            extrinsics = np.eye(4)
            extrinsics[:3, :3] = rotation
            extrinsics[:3, 3] = position
            poses.append(extrinsics)

            # extrinsics = np.dot(M, extrinsics)


            # extrinsics[:3, 2] *= -1.

            # extrinsics = np.linalg.inv(extrinsics)

        return poses


if __name__ == "__main__":
    frame = 80
    dataset = 1
    color_name = '/home/yan/Dataset/ICL/living' + str(dataset) +'/clean/rgb/' + str(frame) +'.png'
    depth_name = '/home/yan/Dataset/ICL/living' + str(dataset) + '/clean/depth/' + str(frame) + '.png'
    pose_name = '/home/yan/Dataset/ICL/living' + str(dataset) +'/clean/traj0.gt.freiburg'
    color_raw = o3d.io.read_image(color_name)#"/home/yan/Dataset/Zhou/burghers/color/001001.png")#"/home/yan/Dataset/ICL/living0/clean/rgb/0.png")#/home/yan/Dataset/TUM/rgbd_dataset_freiburg3_long_office_household/rgb/1341847980.722988.png")#/home/yan/Dataset/ICL/living0/clean/rgb/0.png")
    depth_raw = o3d.io.read_image(depth_name)#"/home/yan/Dataset/Zhou/burghers/depth/001001.png")#"/home/yan/Dataset/ICL/living0/clean/depth/0.png")#/home/yan/Dataset/TUM/rgbd_dataset_freiburg3_long_office_household/depth/1341847980.723020.png")#/home/yan/Dataset/ICL/living0/clean/depth/0.png")
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw,depth_scale=5000.0, depth_trunc=10.0, convert_rgb_to_intensity=False)
    trajectory = read_camera_poses(pose_name)
    print("frame count: ", len(trajectory))
    intrinsic = np.asarray([[481.20, 0., 319.50],
                            [0., -480.05, 239.50],
                            [0., 0., 1.]])
    #kt0
    T0 = np.array([[0.999759, -0.000287637, 0.0219655, -1.36022],
                   [0.000160294, 0.999983, 0.00579897, 1.48382],
                   [0.0219668, 0.00579404, -0.999742, 1.44256],
                   [0, 0, 0, 1]])
    #kt1
    T1 = np.array([[0.99975, -0.00789018, 0.0209474, -0.0976324],
                   [0.00789931, 0.999969, -0.000353282, 1.27618],
                   [0.0209439, -0.000518671, -0.999781, 0.0983734],
                   [0, 0, 0, 1]])
    #kt2
    T2 = np.array([[0.999822, 0.0034419, 0.0185526, -0.786316],
                   [-0.00350915, 0.999987, 0.00359374, 1.28433],
                   [0.01854, 0.00365819, -0.999821, 1.45583],
                   [0, 0, 0, 1]])
    #kt3
    T3 = np.array([[0.999778, -0.000715914, 0.0210893, -1.13311],
                   [0.000583688, 0.99998, 0.0062754, 1.26825],
                   [0.0210934, 0.0062617, -0.999758, 3.151866],
                   [0, 0, 0, 1]])
    dataset_transform = [T0, T1, T2, T3]
    T = np.array([[1.000000, 0.000000, 0.000000, 0.000000],
                  [0.000000, 1.000000, 0.000000, 0.000000],
                  [0.000000, 0.000000, 1.000000, -2.250000],
                  [0.000000, 0.000000, 0.000000, 1.000000]])
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(640,480,481.20,-480.05,319.50,239.50))
    #o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    #pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    #print(trajectory[79])
    pcd.transform(trajectory[79])
    #print(dataset_transform[dataset])
    pcd.transform(dataset_transform[dataset])
    pcd.transform(T)

    #pcd.estimate_normals(
    #search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd],point_show_normal=True)
    o3d.io.write_point_cloud("ICL_clean_80.ply", pcd)
    

    
