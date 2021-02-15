import numpy as np
import open3d as o3d
import os
import pyquaternion
import math

def normalize(vector):
    """Normalize a vector, i.e. make the norm 1 while preserving direction."""
    return vector / np.linalg.norm(vector)


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

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
    #frame = 80
    subsample = 10
    dataset = 2
    folder_name = '/home/yan/Dataset/ICL/living'
    subfolder = 'clean'
    output_dir = 'data/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    count = 0
    for fn in os.listdir(folder_name + str(dataset) + '/' + subfolder + '/rgb/'):
        count = count+1
    print(count)
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
    pose_name = folder_name + str(dataset) + '/' + subfolder + '/traj0.gt.freiburg'
    trajectory = read_camera_poses(pose_name)
    print("frame count: ", len(trajectory))
    intrinsic = np.asarray([[481.20, 0., 319.50],
                            [0., -480.05, 239.50],
                            [0., 0., 1.]])
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    view_control = visualizer.get_view_control()
    cam = view_control.convert_to_pinhole_camera_parameters()
    #cam.intrinsic.intrinsic_matrix = intrinsic
    #visualizer.create_window(width=cam.intrinsic.width, height=cam.intrinsic.height)
    # Set default render options
    render_options = visualizer.get_render_option()
    render_options.line_width = 15.0
    render_options.point_size = 0.1
    render_options.show_coordinate_frame = True
    geometry = o3d.geometry.PointCloud()
    for sample in range(1,count//subsample):
        frame = sample * subsample
        color_name = folder_name + str(dataset) + '/' + subfolder + '/rgb/' + str(frame) +'.png'
        depth_name = folder_name + str(dataset) + '/' + subfolder + '/depth/' + str(frame) + '.png'
        color_raw = o3d.io.read_image(color_name)
        depth_raw = o3d.io.read_image(depth_name)
        rgbd_image_raw = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw,depth_scale=5000.0, depth_trunc=10.0, convert_rgb_to_intensity=False)
    

        pcd_raw = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image_raw,
            o3d.camera.PinholeCameraIntrinsic(640,480,481.20,-480.05,319.50,239.50))
        pose = trajectory[frame-1]
        euler = rotationMatrixToEulerAngles(pose[:3, :3])
        pcd_raw.transform(trajectory[frame-1])
        pcd_raw.transform(dataset_transform[dataset])
        pcd_raw.transform(T)
        pcd_raw.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd_raw.orient_normals_towards_camera_location()
        geometry += pcd_raw
        # # geometry.points += pcd_raw.points
        # # geometry.colors += pcd_raw.colors
        # if sample == 1:
        #     visualizer.add_geometry(geometry)
        # else:
        #     visualizer.update_geometry(geometry)
        visualizer.add_geometry(pcd_raw)

        front       = np.asarray((0,0,1))
        lookat      = normalize(np.asarray((1.0,3.0,1.0)))
        zoom = -0.15
        rTl = np.identity(4)
        rTl[1,1] = -1
        cam.extrinsic = np.dot(np.dot(rTl, pose),np.linalg.inv(rTl))
        view_control.convert_from_pinhole_camera_parameters(cam)
        view_control.set_lookat(lookat)
        view_control.set_zoom(zoom)
        visualizer.poll_events()
        visualizer.update_renderer()
        #visualizer.run()
        #visualizer.destroy_window()
    o3d.io.write_point_cloud('/home/yan/Desktop/ICL2_seq_1-10.xyzn', geometry)
    os.rename('/home/yan/Desktop/ICL2_seq_1-10.xyzn', '/home/yan/Desktop/ICL2_seq_1-10.xyz')


        #o3d.visualization.draw_geometries([pcd_raw],point_show_normal=True)
    

    
