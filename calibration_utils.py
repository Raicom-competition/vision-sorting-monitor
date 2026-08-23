import math

import cv2
import numpy as np


def load_transformation(path):
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    node = storage.getNode("Transformation_Matrix")
    if node.empty():
        storage.release()
        raise ValueError("标定文件中没有 Transformation_Matrix")
    matrix = node.mat()
    storage.release()
    return np.asarray(matrix, dtype=np.float64)


def pose_to_matrix(pose):
    x, y, z, rx, ry, rz = [float(value) for value in pose]
    rx, ry, rz = map(math.radians, [rx, ry, rz])
    r_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(rx), -math.sin(rx)],
            [0, math.sin(rx), math.cos(rx)],
        ]
    )
    r_y = np.array(
        [
            [math.cos(ry), 0, math.sin(ry)],
            [0, 1, 0],
            [-math.sin(ry), 0, math.cos(ry)],
        ]
    )
    r_z = np.array(
        [
            [math.cos(rz), -math.sin(rz), 0],
            [math.sin(rz), math.cos(rz), 0],
            [0, 0, 1],
        ]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = r_z @ r_y @ r_x
    matrix[:3, 3] = [x, y, z]
    return matrix


def camera_point_from_pixel(x, y, depth_m, intrinsics):
    z = depth_m
    x_cam = (x - intrinsics.ppx) * z / intrinsics.fx
    y_cam = (y - intrinsics.ppy) * z / intrinsics.fy
    return np.array([x_cam, y_cam, z, 1.0], dtype=np.float64)


def camera_to_base(point_camera, matrix_cam2grip, matrix_gripper2base):
    point_base = matrix_gripper2base @ matrix_cam2grip @ point_camera
    return point_base[:3]
