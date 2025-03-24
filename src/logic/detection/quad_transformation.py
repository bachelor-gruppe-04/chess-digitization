import numpy as np
from scipy.spatial import Delaunay

x = list(range(7))
y = list(range(7))
GRID = [[(xx, yy) for xx in x] for yy in y]
IDEAL_QUAD = [[0, 1], [1, 1], [1, 0], [0, 0]]

def score_quad(quad, x_corners):
    """
    Scores a quadrilateral by warping it to fit the ideal quad and finding the offset.
    
    @param quad: List of 4 points representing the corners of the quadrilateral.
    @param x_corners: List of all corner points detected from the grid.
    
    @return: A tuple containing the score, transformation matrix, and best offset.
    """
    perspective_matrix = get_perspective_transform(IDEAL_QUAD, quad)
    warped_x_corners = perspective_transform(x_corners, perspective_matrix)
    offset = find_offset(warped_x_corners)

    score = calculate_offset_score(warped_x_corners, offset)
    return score, perspective_matrix, offset

def find_offset(warped_xcorners):
    """
    Finds the best offset to align the warped grid with the original grid.
    
    @param warped_xcorners: List of corner points after perspective transformation.
    
    @return: A list representing the best offset in the x and y directions.
    """
    best_offset = [0, 0]
    for i in range(2):  # Iterate over x and y dimensions
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
    """
    Computes the pairwise Euclidean distance between two sets of points.
    
    @param a: List of points (x, y) in the first set.
    @param b: List of points (x, y) in the second set.
    
    @return: A numpy array of distances between each pair of points.
    """
    a = np.array(a)
    b = np.array(b)
    dist = np.sqrt(np.sum((a[:, np.newaxis, :] - b[np.newaxis, :, :])**2, axis=2))
    return dist

def calculate_offset_score(warped_x_corners, shift):
    """
    Calculates the alignment score for the warped grid with a given shift.
    
    @param warped_x_corners: List of points after perspective transformation.
    @param shift: Offset to apply to the grid.
    
    @return: A score representing how well the warped grid aligns with the shifted grid.
    """
    grid = np.array([[x[0] + shift[0], x[1] + shift[1]] for sublist in GRID for x in sublist])
    dist = cdist(grid, warped_x_corners)
    assignment_cost = sum(min(dist[i]) for i in range(len(dist)))  # Total cost of closest point assignment
    score = 1 / (1 + assignment_cost)  # Lower cost results in a higher score
    return score

def get_perspective_transform(target, keypoints):
    A = np.zeros((8, 8))
    B = np.zeros((8, 1))

    for i in range(4):
        x, y = keypoints[i]
        u, v = target[i]
        
        A[i * 2, 0] = x
        A[i * 2, 1] = y
        A[i * 2, 2] = 1
        A[i * 2, 6] = -u * x
        A[i * 2, 7] = -u * y
        
        A[i * 2 + 1, 3] = x
        A[i * 2 + 1, 4] = y
        A[i * 2 + 1, 5] = 1
        A[i * 2 + 1, 6] = -v * x
        A[i * 2 + 1, 7] = -v * y
        
        B[i * 2, 0] = u
        B[i * 2 + 1, 0] = v

    # Solving for the transform matrix
    solution = np.linalg.solve(A, B)
    
    # Adding the last element (1) to form a 3x3 matrix
    transform = np.append(solution, 1).reshape((3, 3))

    return transform



def perspective_transform(src, transform):
    """
    Applies a perspective transformation to a set of points.
    
    @param src: List of points (x, y) to be transformed.
    @param transform: 3x3 transformation matrix.
    
    @return: List of transformed points (x', y') after perspective division.
    """
    if len(src[0]) == 2:
        src = [[x[0], x[1], 1] for x in src]
    
    src_array = np.array(src)
    warped_src = np.dot(src_array, transform.T)
    
    for i in range(warped_src.shape[0]):
        x = warped_src[i, 0]
        y = warped_src[i, 1]
        w = warped_src[i, 2]
        warped_src[i, 0] = x / w
        warped_src[i, 1] = y / w
    
    warped_src_array = warped_src[:, :2]
    return warped_src_array.tolist()


def get_quads(x_corners):
    """
    Finds quads from a set of x_corners using Delaunay triangulation.

    Args:
        x_corners: A list of lists, where each inner list represents an (x, y) coordinate.

    Returns:
        A list of lists of lists, where each inner list of lists represents a quad.
    """
    points = np.array(x_corners)
    delaunay = Delaunay(points)
    triangles = delaunay.simplices
    quads = []

    for i in range(0, len(triangles), 1):
        t1, t2, t3 = triangles[i]
        quad = [t1, t2, t3, -1]

        for j in range(0, len(triangles), 1):
            if i == j:
                continue

            j_tri = triangles[j]
            cond1 = (t1 in j_tri and t2 in j_tri)
            cond2 = (t2 in j_tri and t3 in j_tri)
            cond3 = (t3 in j_tri and t1 in j_tri)

            if cond1 or cond2 or cond3:
                # find the other point
                other_point = None
                for point in j_tri:
                    if point not in quad:
                        other_point = point
                        break
                if other_point is not None:
                  quad[3] = other_point
                  break

        if quad[3] != -1:
            quads.append([x_corners[x] for x in quad])

    return quads

