import numpy as np


def save_cam_poses_24():
    cam_poses = []
    obj_poses = []

    dist = 1.1
    elevs = np.radians([15, 40, 65] * 8)
    azims = np.radians(np.repeat(np.arange(0, 360, 45), 3))
    
    for i, (elev, azim) in enumerate(list(zip(elevs, azims))):
        cam_pose = np.eye(4)
        y = dist * np.sin(elev)
        x = dist * np.cos(elev) * np.sin(azim)
        z = dist * np.cos(elev) * np.cos(azim)
        cam_pose[:3, 3] = [x, y, z]
        rotx = np.array([
            [1.0, 0, 0],
            [0.0, np.cos(elev), np.sin(elev)],
            [0.0, -np.sin(elev), np.cos(elev)]
        ])
        roty = np.array([
            [np.cos(azim), 0, np.sin(azim)],
            [0.0, 1, 0.0],
            [-np.sin(azim), 0, np.cos(azim)]
        ])
        cam_pose[:3, :3] = np.matmul(roty, rotx)
        world2cam = np.linalg.inv(cam_pose)
        
        cam_poses.append(cam_pose[None, ...])
        obj_poses.append(world2cam[None, ...])

    cam_poses = np.concatenate(cam_poses, axis=0)
    obj_poses = np.concatenate(obj_poses, axis=0)

    np.save("data/renders-poses/cam_poses_24.npy", cam_poses)
    np.save("data/renders-poses/obj_poses_24.npy", obj_poses)


def save_cam_poses_60_wss():
    cam_poses = []
    obj_poses = []

    dist = 1.1
    elevs = np.radians([0, 15, 30, 45, 60] * 12)
    azims = np.radians(np.repeat(np.arange(0, 360, 30), 5))
    # elevs = np.radians([60, 30, 0, -30, -60] * 12) # 30, 60, 90, 120, 150
    # azims = np.radians(np.repeat(np.arange(0, 360, 30), 5))
    
    for i, (elev, azim) in enumerate(list(zip(elevs, azims))):
        cam_pose = np.eye(4)
        # elev = np.pi/2 - elev
        y = dist * np.sin(elev)
        x = dist * np.cos(elev) * np.sin(azim)
        z = dist * np.cos(elev) * np.cos(azim)
        cam_pose[:3, 3] = [x, y, z]
        rotx = np.array([
            [1.0, 0, 0],
            [0.0, np.cos(elev), np.sin(elev)],
            [0.0, -np.sin(elev), np.cos(elev)]
        ])
        roty = np.array([
            [np.cos(azim), 0, np.sin(azim)],
            [0.0, 1, 0.0],
            [-np.sin(azim), 0, np.cos(azim)]
        ])
        cam_pose[:3, :3] = np.matmul(roty, rotx)
        world2cam = np.linalg.inv(cam_pose)
        
        cam_poses.append(cam_pose[None, ...])
        obj_poses.append(world2cam[None, ...])

    cam_poses = np.concatenate(cam_poses, axis=0)
    obj_poses = np.concatenate(obj_poses, axis=0)

    np.save("data/renders-poses/cam_poses_60_wss.npy", cam_poses)
    np.save("data/renders-poses/obj_poses_60_wss.npy", obj_poses)


def save_cam_poses_180_wss():
    cam_poses = []
    obj_poses = []

    dist = 1.1
    elevs = np.radians([0, 15, 30, 45, 60] * 36)
    azims = np.radians(np.repeat(np.arange(0, 360, 10), 5))
    
    for i, (elev, azim) in enumerate(list(zip(elevs, azims))):
        cam_pose = np.eye(4)
        y = dist * np.sin(elev)
        x = dist * np.cos(elev) * np.sin(azim)
        z = dist * np.cos(elev) * np.cos(azim)
        cam_pose[:3, 3] = [x, y, z]
        rotx = np.array([
            [1.0, 0, 0],
            [0.0, np.cos(elev), np.sin(elev)],
            [0.0, -np.sin(elev), np.cos(elev)]
        ])
        roty = np.array([
            [np.cos(azim), 0, np.sin(azim)],
            [0.0, 1, 0.0],
            [-np.sin(azim), 0, np.cos(azim)]
        ])
        cam_pose[:3, :3] = np.matmul(roty, rotx)
        world2cam = np.linalg.inv(cam_pose)
        
        cam_poses.append(cam_pose[None, ...])
        obj_poses.append(world2cam[None, ...])

    cam_poses = np.concatenate(cam_poses, axis=0)
    obj_poses = np.concatenate(obj_poses, axis=0)

    np.save("data/renders-poses/cam_poses_180_wss.npy", cam_poses)
    np.save("data/renders-poses/obj_poses_180_wss.npy", obj_poses)


def save_cam_poses_60_shapenet():
    cam_poses = []
    obj_poses = []

    dist = 1.1
    elevs = np.radians([30, 60, 90, 120, 150] * 12)
    azims = np.radians(np.repeat(np.arange(0, 360, 360 // 12), 5))
    
    for i, (elev, azim) in enumerate(list(zip(elevs, azims))):
        cam_pose = np.eye(4)
        elev = np.pi/2 - elev
        azim = azim + np.pi
        y = dist * np.sin(elev)
        x = dist * np.cos(elev) * np.sin(azim)
        z = dist * np.cos(elev) * np.cos(azim)
        cam_pose[:3, 3] = [x, y, z]
        rotx = np.array([
            [1.0, 0, 0],
            [0.0, np.cos(elev), np.sin(elev)],
            [0.0, -np.sin(elev), np.cos(elev)]
        ])
        roty = np.array([
            [np.cos(azim), 0, np.sin(azim)],
            [0.0, 1, 0.0],
            [-np.sin(azim), 0, np.cos(azim)]
        ])
        cam_pose[:3, :3] = np.matmul(roty, rotx)
        world2cam = np.linalg.inv(cam_pose)
        
        cam_poses.append(cam_pose[None, ...])
        obj_poses.append(world2cam[None, ...])

    cam_poses = np.concatenate(cam_poses, axis=0)
    obj_poses = np.concatenate(obj_poses, axis=0)

    np.save("data/renders-poses/cam_poses_60_shapenet.npy", cam_poses)
    np.save("data/renders-poses/obj_poses_60_shapenet.npy", obj_poses)


def save_cam_poses_180_shapenet():
    cam_poses = []
    obj_poses = []

    dist = 1.1
    elevs = np.radians([0, 15, 30, 45, 60] * 36)
    azims = np.radians(np.repeat(np.arange(0, 360, 10), 5))
    
    for i, (elev, azim) in enumerate(list(zip(elevs, azims))):
        cam_pose = np.eye(4)
        azim = azim + np.pi
        y = dist * np.sin(elev)
        x = dist * np.cos(elev) * np.sin(azim)
        z = dist * np.cos(elev) * np.cos(azim)
        cam_pose[:3, 3] = [x, y, z]
        rotx = np.array([
            [1.0, 0, 0],
            [0.0, np.cos(elev), np.sin(elev)],
            [0.0, -np.sin(elev), np.cos(elev)]
        ])
        roty = np.array([
            [np.cos(azim), 0, np.sin(azim)],
            [0.0, 1, 0.0],
            [-np.sin(azim), 0, np.cos(azim)]
        ])
        cam_pose[:3, :3] = np.matmul(roty, rotx)
        world2cam = np.linalg.inv(cam_pose)
        
        cam_poses.append(cam_pose[None, ...])
        obj_poses.append(world2cam[None, ...])

    cam_poses = np.concatenate(cam_poses, axis=0)
    obj_poses = np.concatenate(obj_poses, axis=0)

    np.save("data/renders-poses/cam_poses_180_shapenet.npy", cam_poses)
    np.save("data/renders-poses/obj_poses_180_shapenet.npy", obj_poses)


save_cam_poses_24()
save_cam_poses_60_wss()
save_cam_poses_180_wss()

save_cam_poses_60_shapenet()
save_cam_poses_180_shapenet()
