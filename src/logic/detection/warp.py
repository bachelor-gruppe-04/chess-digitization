import numpy as np

from constants import SQUARE_SIZE, BOARD_SIZE


def perspective_transform(src, transform):
    if len(src[0]) == 2:
        src = [x + [1] for x in src]
    
    warped_src = np.dot(src, transform.T)
    warped_src /= warped_src[:, 2].reshape(-1, 1)
    return warped_src[:, :2].tolist()


def get_perspective_transform(target, keypoints):
    A = np.zeros((8, 8))
    B = np.zeros((8, 1))
    
    for i in range(4):
        x, y = keypoints[i]
        u, v = target[i]
        A[i * 2] = [x, y, 1, 0, 0, 0, -u * x, -u * y]
        A[i * 2 + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]
        B[i * 2] = u
        B[i * 2 + 1] = v
    
    solution = np.linalg.solve(A, B).flatten()
    transform = np.vstack([solution.reshape(8, 1), [[1]]]).reshape(3, 3)
    return transform


def get_inv_transform(keypoints):
    target = [
        [BOARD_SIZE, BOARD_SIZE],
        [0, BOARD_SIZE],
        [0, 0],
        [BOARD_SIZE, 0]
    ]
    transform = get_perspective_transform(target, keypoints)
    return np.linalg.inv(transform)


def transform_centers(inv_transform):
    x = [0.5 + i for i in range(8)]
    y = [7.5 - i for i in range(8)]
    warped_centers = [[xx * SQUARE_SIZE, yy * SQUARE_SIZE, 1] for yy in y for xx in x]
    centers = perspective_transform(warped_centers, inv_transform)
    centers3D = np.expand_dims(np.array(centers), axis=0)
    return centers, centers3D


def transform_boundary(inv_transform):
    warped_boundary = [
        [-0.5 * SQUARE_SIZE, -0.5 * SQUARE_SIZE, 1],
        [-0.5 * SQUARE_SIZE, 8.5 * SQUARE_SIZE, 1],
        [8.5 * SQUARE_SIZE, 8.5 * SQUARE_SIZE, 1],
        [8.5 * SQUARE_SIZE, -0.5 * SQUARE_SIZE, 1]
    ]
    boundary = perspective_transform(warped_boundary, inv_transform)
    boundary3D = np.expand_dims(np.array(boundary), axis=0)
    return boundary, boundary3D
