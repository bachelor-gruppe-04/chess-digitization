import numpy as np


x = list(range(7))
y = list(range(7))
GRID = [[(xx, yy) for xx in x] for yy in y]
IDEAL_QUAD = [[0, 1], [1, 1], [1, 0], [0, 0]]



def score_quad(quad, x_corners):
    M = get_perspective_transform(IDEAL_QUAD, quad)
    warped_x_corners = perspective_transform(x_corners, M)
    offset = find_offset(warped_x_corners)

    score = calculate_offset_score(warped_x_corners, offset)
    return score, M, offset


def find_offset(warped_xcorners):
    best_offset = [0, 0]
    for i in range(2):
        low = -7
        high = 1
        scores = {}
        while (high - low) > 1:
            mid = (high + low) // 2
            for x in [mid, mid + 1]:
                if x not in scores:
                    shift = [0, 0]
                    shift[i] = x
                    scores[x] = calculate_offset_score(warped_xcorners, shift)
            if scores[mid] > scores[mid + 1]:
                high = mid
            else:
                low = mid
        best_offset[i] = low + 1

    return best_offset


def cdist(a, b):
    # Convert lists to numpy arrays for efficient computation
    a = np.array(a)
    b = np.array(b)

    # Compute the pairwise Euclidean distance
    dist = np.sqrt(np.sum((a[:, np.newaxis, :] - b[np.newaxis, :, :])**2, axis=2))

    return dist


def calculate_offset_score(warped_x_corners, shift):
    grid = np.array([ [x[0] + shift[0], x[1] + shift[1]] for x in GRID ])
    dist = cdist(grid, warped_x_corners)

    assignment_cost = 0
    for i in range(len(dist)):
        assignment_cost += min(dist[i])

    score = 1 / (1 + assignment_cost)

    return score


def get_perspective_transform(target, keypoints):
    # Create the matrix A and vector B
    A = np.zeros((8, 8))
    B = np.zeros((8, 1))

    for i in range(4):
        x, y = keypoints[i]
        u, v = target[i]
        
        A[i*2, 0] = x
        A[i*2, 1] = y
        A[i*2, 2] = 1
        A[i*2, 6] = -u * x
        A[i*2, 7] = -u * y
        
        A[i*2 + 1, 3] = x
        A[i*2 + 1, 4] = y
        A[i*2 + 1, 5] = 1
        A[i*2 + 1, 6] = -v * x
        A[i*2 + 1, 7] = -v * y
        
        B[i*2, 0] = u
        B[i*2 + 1, 0] = v

    # Solve the linear system A * M = B
    solution = np.linalg.solve(A, B)

    # Reshape the solution to form a 3x3 transformation matrix
    transform = np.hstack((solution, [[1]]))  # Adding the last element '1' to make it 3x3
    transform = transform.reshape((3, 3))

    return transform



def perspective_transform(src, transform):
    # Convert 2D points to homogeneous coordinates if necessary
    if len(src[0]) == 2:
        src = [[x[0], x[1], 1] for x in src]
    
    # Convert the list to a numpy array
    src_array = np.array(src)
    
    # Apply the transformation (matrix multiplication)
    warped_src = np.dot(src_array, transform.T)
    
    # Perform perspective division
    for i in range(warped_src.shape[0]):
        x = warped_src[i, 0]
        y = warped_src[i, 1]
        w = warped_src[i, 2]
        warped_src[i, 0] = x / w
        warped_src[i, 1] = y / w
    
    # Extract the 2D coordinates and return
    warped_src_array = warped_src[:, :2]
    
    return warped_src_array.tolist()



def get_quads(x_corners):
    # Flatten and round the coordinates
    int_x_corners = np.round(np.array(x_corners).flatten()).astype(int)
    
    # Create the Delaunay triangulation
    points = np.array(x_corners)
    delaunay = Delaunay(points)
    triangles = delaunay.simplices

    quads = []
    
    # Loop over the triangles
    for i in range(0, len(triangles), 3):
        t1, t2, t3 = triangles[i]
        quad = [t1, t2, t3, -1]

        # Search for the corresponding quadrilateral
        for j in range(0, len(triangles), 3):
            if i == j:
                continue
            cond1 = (t1 == triangles[j][0] and t2 == triangles[j][1]) or (t1 == triangles[j][1] and t2 == triangles[j][0])
            cond2 = (t2 == triangles[j][0] and t3 == triangles[j][1]) or (t2 == triangles[j][1] and t3 == triangles[j][0])
            cond3 = (t3 == triangles[j][0] and t1 == triangles[j][1]) or (t3 == triangles[j][1] and t1 == triangles[j][0])
            if cond1 or cond2 or cond3:
                quad[3] = triangles[j][2]
                break

        # Add the quad to the list if it's valid
        if quad[3] != -1:
            quads.append([x_corners[x] for x in quad])

    return quads



