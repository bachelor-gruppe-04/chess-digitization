import numpy as np
import tensorflow as tf

from typing import List, Tuple
from logic.machine_learning.utilities.constants import SQUARE_SIZE, BOARD_SIZE

def perspective_transform(src: List[List[float]], transform: np.ndarray) -> List[List[float]]:
    """
    Applies a perspective transformation to a set of 2D points using a 3x3 transformation matrix.

    This function takes a list or array of 2D points (in Cartesian coordinates), 
    converts them to homogeneous coordinates (3D) (if necessary), applies the perspective 
    transformation using the provided matrix, and then returns the transformed points 
    in 2D space after normalizing back from homogeneous coordinates (3D).

    Args:
        src (list of list of floats): A list of 2D points (each as [x, y]) to be transformed.
        transform (np.ndarray): A 3x3 perspective transformation matrix that is used to transform the points.

    Returns:
        list: A list of transformed 2D points after applying the perspective transformation.
    """
    
    # Step 1: Check if the input points are 2D (i.e., [x, y]). 
    # If so, add a homogeneous coordinate (1) to make them 3D points [x, y, 1].
    if len(src[0]) == 2:
        src = [x + [1] for x in src]

    # Step 2: Apply the perspective transformation using matrix multiplication.
    # The 'src' points are multiplied with the transpose of the transformation matrix.
    warped_src = np.dot(src, transform.T)
    # The result of np.dot is in homogeneous coordinates (x', y', w).

    # Step 3: Normalize the transformed points by dividing by the 'w' coordinate.
    # This step converts the points back to Cartesian coordinates.
    warped_src /= warped_src[:, 2].reshape(-1, 1) 
    # We divide each point's [x', y', w] by w to normalize and get [x' / w, y' / w, 1].

    # Step 4: Return the 2D transformed points (excluding the homogeneous coordinate).
    return warped_src[:, :2].tolist()  



def get_perspective_transform(target: List[Tuple[float, float]], keypoints: List[Tuple[float, float]]) -> np.ndarray:
    """
    Computes the perspective transformation matrix that maps the xy in `keypoints` to the xy in `target`.

    This function uses the method of solving a system of linear equations derived from the perspective transformation.
    
    Args:
    - target (list of tuples): The target points that we want to map the keypoints to. 
      These are usually the coordinates of a square or a rectangle.
    - keypoints (list of tuples): The source points that we are transforming from. These could be the corners of a chessboard or another object.

    Returns:
    - numpy.ndarray: The 3x3 perspective transformation matrix.
    
    The perspective transformation matrix allows for transforming points from the `keypoints` plane to the `target` plane.
    """
    
    # Initialize matrix A (coefficients) and vector B (constants) for the system of linear equations
    A = np.zeros((8, 8))  
    B = np.zeros((8, 1))  

    # For each of the four points, fill in the corresponding row of A and B
    for i in range(4):
        x, y = keypoints[i]   
        u, v = target[i]     

        # Populate the matrix A based on the equations derived from the perspective transformation
        A[i * 2] = [x, y, 1, 0, 0, 0, -u * x, -u * y]   # First equation (mapping x to u)
        A[i * 2 + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]   # Second equation (mapping y to v)

        # Populate the constants vector B with the target coordinates
        B[i * 2] = u
        B[i * 2 + 1] = v
    
    # Solve the system of equations A * solution = B using numpy's linear solver
    # This returns the transformation parameters (flattened into a 1D array)
    solution = np.linalg.solve(A, B).flatten() 

    # Reshape the solution into a 3x3 matrix (perspective transformation matrix)
    transform = np.vstack([solution.reshape(8, 1), [[1]]]).reshape(3, 3)

    return transform



def get_inv_transform(keypoints: List[Tuple[float, float]]) -> np.ndarray:
    """
    Computes the inverse of the perspective transformation matrix that maps the keypoints 
    to a square with a defined size (usually for tasks like warping images or correcting perspective distortion).
    
    This function calculates the perspective transformation matrix using the provided keypoints 
    and then returns its inverse.

    Why do we need to define a target?
    ----------------------------------
    - The transformation process does not inherently know what the correct shape of the object should be.
    - We need to explicitly define a target shape (in this case, a perfect square) so that the perspective transformation 
      knows what form the keypoints should be mapped into.
    - If we didn't specify a target, the transformation could map our points into any arbitrary shape instead of straightening the board.

    Why do we compute the inverse?
    -----------------------------
    - The perspective transformation matrix typically maps points from the keypoints to the target.
    - However, in many applications (such as unwarping images), we need to reverse this transformation.
    - To go from the target (square) back to the original keypoints, we compute the inverse of the transformation matrix.
    - Mathematically, if `T` is the transformation matrix that maps `keypoints → target`, then `T⁻¹` allows us to do the reverse:  
      ```
      keypoints = T * target
      target = T⁻¹ * keypoints
      ```
    
    - T distorts the points (maps them from the perfect square to the distorted shape).
    - T⁻¹ undistorts the points (maps them from the distorted shape back to the perfect square).

    Args:
    - keypoints (list of tuples): The four corner points of the real-world object (chessboard). These points may form a trapezoid rather than a perfect square.

    Returns:
    - numpy.ndarray: The inverse of the 3x3 perspective transformation matrix.
    
    This matrix is used to transform points in the reverse direction (from the perfect square back to the distorted input shape).
    """

    # Define the target points (the desired output coordinates of the perspective transform)
    # Since we want the transformed chessboard to be a perfect square, we define these four corners:
    target: List[Tuple[float, float]] = [
        [BOARD_SIZE, BOARD_SIZE],  # Top-right corner of the square
        [0, BOARD_SIZE],           # Top-left corner of the square
        [0, 0],                    # Bottom-left corner of the square
        [BOARD_SIZE, 0]            # Bottom-right corner of the square
    ]
    
    # Compute the transformation matrix (3x3) that maps the given keypoints to this target square
    transform: np.ndarray = get_perspective_transform(target, keypoints)
    
    # Compute the inverse of the transformation matrix to reverse the transformation
    inv_transform: np.ndarray = np.linalg.inv(transform)
    
    return inv_transform



def transform_centers(inv_transform: np.ndarray) -> List[List[float]]:
    """
    Transforms the centers of an 8x8 chessboard grid from their positions in a perfect square 
    to their corresponding positions in the distorted image using the provided inverse perspective transformation.

    Since we're working with a perfect square grid, the centers fall on evenly spaced values 
    from `0.5` to `7.5` in both x and y directions. These values are multiplied by `SQUARE_SIZE` 
    to get the pixel coordinates.

    Args:
    - inv_transform (numpy.ndarray): The inverse perspective transformation matrix.

    Returns:
    - centers (list): The transformed center coordinates in the distorted space.
    """
    x: List[float] = [0.5 + i for i in range(8)]
    y: List[float] = [7.5 - i for i in range(8)]
    centers_in_perfect_square: List[List[float]] = [[xx * SQUARE_SIZE, yy * SQUARE_SIZE, 1] for yy in y for xx in x]
    
    centers: List[List[float]] = perspective_transform(centers_in_perfect_square, inv_transform)    
    centers3D = np.expand_dims(np.array(centers), axis=0)
    
    return centers, centers3D



def transform_boundary(inv_transform: np.ndarray):
    """
    Transforms the boundary of an 8x8 chessboard grid from its ideal square representation 
    to its corresponding distorted representation in the input image using the inverse 
    perspective transformation matrix.

    The function defines a square boundary around the chessboard in the perfect grid space 
    (with slight padding around the board edges), then applies the inverse transform to map 
    this boundary into the distorted input image coordinates.

    Args:
    - inv_transform (numpy.ndarray): The inverse perspective transformation matrix 
      that maps points from the ideal (square) grid space back to the distorted input space.

    Returns:
    - boundary (List[List[float]]): A list of 4 points (in [x, y]) representing the transformed boundary 
      corners of the chessboard in the distorted space.
    - boundary3D (tf.Tensor): A 3D tensor (shape: [1, 4, 2]) containing the same boundary points 
      as a TensorFlow tensor, which can be directly used for further processing (e.g., masks or overlays).
    """

    # Define a slightly expanded square around the 8x8 grid in perfect square space
    warped_boundary = np.array([
        [-0.5 * SQUARE_SIZE, -0.5 * SQUARE_SIZE, 1],   # Top-left (outside the board)
        [-0.5 * SQUARE_SIZE, 8.5 * SQUARE_SIZE, 1],    # Bottom-left
        [8.5 * SQUARE_SIZE, 8.5 * SQUARE_SIZE, 1],     # Bottom-right
        [8.5 * SQUARE_SIZE, -0.5 * SQUARE_SIZE, 1]     # Top-right
    ])

    # Apply inverse perspective transformation to map to distorted input space
    boundary = perspective_transform(warped_boundary, inv_transform)

    # Convert to a TensorFlow 3D tensor for further operations (e.g., masking in TF)
    boundary3D = tf.expand_dims(tf.convert_to_tensor(boundary, dtype=tf.float32), axis=0)

    return boundary, boundary3D
