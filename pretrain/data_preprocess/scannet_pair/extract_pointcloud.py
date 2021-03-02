import open3d as o3d
import cv2
import sys
import numpy as np
from glob import glob
import os

def main():
    nump = 50000 # number of points from the depth scans
    scenelist = glob('/home/data/scannet/scans/*')
    datalist = []
    for scene in scenelist:
        scene = scene + '/depth'
        framelist = glob(scene+'/*') #home/data/scannet/scans/scene/depth
        intrinsic = np.loadtxt(scene.replace("depth", "intrinsic")+"/intrinsic_depth.txt")
        intrin_cam = o3d.camera.PinholeCameraIntrinsic()
        print(scene)
        if not os.path.exists(scene.replace("depth", "pc")):
            os.mkdir(scene.replace("depth", "pc"))
        if not os.path.exists(scene.replace("depth", "npy")):
                os.mkdir(scene.replace("depth", "npy"))


        for fileidx in range(len(framelist)):
            frame = scene + "/%d.png" % fileidx
            depth = frame
            rgbpath = frame.replace("depth", "color").replace(".png", ".jpg")

            depth_im = cv2.imread(depth, -1)

            try:
                o3d_depth = o3d.geometry.Image(depth_im)
                rgb_im = cv2.resize(cv2.imread(rgbpath), (depth_im.shape[1], depth_im.shape[0]))
                o3d_rgb = o3d.geometry.Image(rgb_im)
                o3d_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_rgb, o3d_depth, depth_scale = 1000.0, depth_trunc = 1000.0, convert_rgb_to_intensity = False)
            except:
                print(frame)
                continue

            intrin_cam.set_intrinsics(width=depth_im.shape[1], height=depth_im.shape[0], fx=intrinsic[1,1], fy=intrinsic[0,0], cx=intrinsic[1,2], cy=intrinsic[0,2])
            pts = o3d.geometry.PointCloud.create_from_rgbd_image(o3d_rgbd, intrin_cam, np.eye(4))

            if len(np.array(pts.points)) > nump:
                sel_idx = np.random.choice(len(np.array(pts.points)), nump, replace=False)
            else:
                sel_idx = np.random.choice(len(np.array(pts.points)), nump, replace=True)
            temp = np.array(pts.points)[sel_idx]

            color_points = np.array(pts.colors)[sel_idx]
            color_points[:, [0,1,2]] = color_points[:, [2,1,0]]
            
            pts.points = o3d.utility.Vector3dVector(temp)
            pts.colors = o3d.utility.Vector3dVector(color_points)
            data = np.concatenate([temp, color_points], axis=1)

            o3d.io.write_point_cloud(scene.replace("depth", "pc")+"/"+frame.split("/")[-1].split(".")[0] + ".ply", pts)
            np.save(scene.replace("depth", "npy")+"/"+frame.split("/")[-1].split(".")[0] + ".npy", data)
            datalist.append(os.path.abspath(scene.replace("depth", "npy")+"/"+frame.split("/")[-1].split(".")[0]+".npy"))

    np.save("/home/data/PointContrast/datalist.txt", datalist)

if __name__=="__main__":
    main()


    

            
