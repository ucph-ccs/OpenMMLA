import json
import math

import numpy as np
from scipy.spatial.transform import Rotation


def transform_rotation(R, R_alt):
    """Convert the rotation matrix from the alternative coordinate system to the main coordinate system.

    Args:
        R (3x3 list): Rotation matrix to convert from alternative to main coordinate system
        R_alt (3x3 list): Rotation matrix in the alternative coordinate system

    Returns:
        R_transformed (3x3 list): Transformed rotation matrix in the main coordinate system
    """
    R = np.array(R)
    R_alt = np.array(R_alt)
    R_transformed = np.dot(R, R_alt)

    return R_transformed.tolist()


def transform_point(point, R, T):
    """Transform a point from the alternative coordinate system to the main coordinate system.

    Args:
        point (1x3 list): Point in the alternative coordinate system
        R (3x3 list): Rotation matrix to convert from alternative to main
        T (3x1 list): Translation vector to convert from alternative to main

    Returns:
        transformed_point (1x3 list): Transformed point in the main coordinate system
    """
    point = np.array(point)
    R = np.array(R)
    T = np.array(T)
    transformed_point = np.dot(R, point) + T

    return transformed_point.tolist()


def distance_between_points(point1, point2):
    """Calculate the Euclidean distance between two points.

    Args:
        point1 (1x3 list): First point
        point2 (1x3 list): Second point

    Returns:
        distance (float): Euclidean distance between point1 and point2
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    distance = np.linalg.norm(point1 - point2)

    return distance


def distance_between_rotations(P, Q):
    """The distance between rotations represented by rotation matrices and is the angle of the difference rotation
    represented by the rotation matrix R=PQ*.

    ref: http://www.boris-belousov.net/2016/12/01/quat-dist/#using-rotation-matrices

    Args:
        P (3x3 list): Rotation matrix P
        Q (3x3 list): Rotation matrix Q

    Returns:
        distance (float): The distance between the rotations P and Q in degrees
    """
    P = np.array(P)
    Q = np.array(Q)
    R = np.dot(P, Q.T)
    value = np.clip((np.trace(R) - 1) / 2, -1, 1)
    theta = math.acos(value)

    return math.degrees(theta)


def direct_transform_matrices(R_main, T_main, R_alt, T_alt):
    """Find transform matrices that converts alternative rotation and translation to the main coordinate system.

    Args:
        R_main (3x3 list): Main rotation matrix
        T_main (3x1 list): Main translation vector
        R_alt (3x3 list): Alternative rotation matrix
        T_alt (3x1 list): Alternative translation vector

    Returns:
        R (3x3 list): Rotation matrix to convert alternative to main
        T (3x1 list): Translation vector to convert alternative to main
    """
    R_main = np.array(R_main)
    T_main = np.array(T_main)
    R_alt = np.array(R_alt)
    T_alt = np.array(T_alt)

    R = np.dot(R_main, np.linalg.inv(R_alt))
    T = T_main - np.dot(R, T_alt)

    return R.tolist(), T.tolist()


def chain_transform_matrices(rotation_matrices, translation_vectors):
    """Chain multiple transformations to compute the final transformation from the start to the end of the sequence.

    Args:
        rotation_matrices (list of 3x3 list): List of rotation matrices representing transformations.
        translation_vectors (list of 3x1 list): List of translation vectors representing transformations.

    Returns:
        R_final (3x3 list): Final rotation matrix after all transformations.
        T_final (3x1 list): Final translation vector after all transformations.
    """
    if len(rotation_matrices) != len(translation_vectors):
        raise ValueError("The number of rotation matrices and translation vectors must be the same.")

    R_final = np.eye(3)
    T_final = np.array([[0], [0], [0]])
    for R, T in zip(rotation_matrices, translation_vectors):
        R = np.array(R)
        T = np.array(T)
        print(f'R: {R}')
        print(f'T: {T}')
        R_final = np.dot(R, R_final)
        T_final = np.dot(R, T_final) + T

    print(f'R final: {R_final}')
    print(f'T final: {T_final}')

    return R_final.tolist(), T_final.tolist()


def average_transform_matrices(rotation_matrices, translation_matrices):
    """Average the transform matrices.

    Args:
        rotation_matrices (list of 3x3 list): List of rotation matrices
        translation_matrices (list of 3x1 list): List of translation vectors

    Returns:
        avg_rot_mat (3x3 list): Average rotation matrix
        avg_trans_mat (3x1 list): Average translation vector
    """
    rotation_matrices = np.array(rotation_matrices)
    translation_matrices = np.array(translation_matrices)

    print(f'R shape: {rotation_matrices.shape}')
    print(f'T shape: {translation_matrices.shape}')

    # Average the translation vectors and rotation matrices
    avg_trans_mat = np.mean(translation_matrices, axis=0)
    avg_rot_mat = average_rotation_matrices(rotation_matrices)

    return avg_rot_mat.tolist(), avg_trans_mat.tolist()


def average_rotation_matrices(rot_matrices):
    """Calculate the average rotation matrix from a list of rotation matrices.

    Args:
        rot_matrices (np.ndarray): A list of 3x3 rotation matrices.

    Returns:
        np.ndarray: A 3x3 average rotation matrix calculated from the input list of rotation matrices.
    """
    quaternions = []

    # Convert each rotation matrix to quaternion
    for rot_matrix in rot_matrices:
        rot = Rotation.from_matrix(rot_matrix)
        quat = rot.as_quat()
        quaternions.append(quat)

    # Average quaternions and normalize it
    avg_quat = np.mean(quaternions, axis=0)
    avg_quat = avg_quat / np.linalg.norm(avg_quat)

    # Convert back to a rotation matrix
    avg_rot = Rotation.from_quat(avg_quat)
    avg_rot_matrix = avg_rot.as_matrix()

    return avg_rot_matrix


def find_transformation_paths(data, start, end, path=None):
    """Recursively find all possible transformation paths from start to end.

    Args:
        data (dict): Dictionary containing transformation data
        start (str): Starting coordinate system
        end (str): Ending coordinate system
        path (list): List of keys representing the path taken so far

    Returns:
        paths (list): List of all possible transformation paths from start to end

    Example:
        transform_data = {
            'a-m': {'R': [[1]], 'T': [[1]]},  # Direct transformation from 'a' to 'm'
            'b-c': {'R': [[1]], 'T': [[1]]},  # Transformation from 'b' to 'c'
            'c-b': {'R': [[1]], 'T': [[1]]},  # Transformation from 'c' to 'b'
            'b-a': {'R': [[1]], 'T': [[1]]},  # Transformation from 'b' to 'a'
            'c-m': {'R': [[1]], 'T': [[1]]},  # Direct transformation from 'c' to 'm'
            'e-b': {'R': [[1]], 'T': [[1]]}  # Transformation from 'e' to 'b'
        }
        Paths from b to m:
        ['b-c', 'c-b', 'b-a', 'a-m']
        ['b-c', 'c-m']
        ['b-a', 'a-m']
    """
    if path is None:
        path = []
    if start == end:
        return [path]
    else:
        paths = []
        for key in data:
            prefix, suffix = key.split('-')
            if prefix == start and key not in path:
                new_paths = find_transformation_paths(data, suffix, end, path + [key])
                for p in new_paths:
                    paths.append(p)
        return paths


def compute_paths_transformation(data, paths):
    """Compute and average the transformation matrices from the start to the end of the sequence for each path.

    This function calculates the final rotation and translation matrices for each path, then averages these matrices
    to provide a single representative transformation for each starting system.

    Args:
        data (dict): Dictionary containing transformation data.
        paths (list of str list): List of all possible transformation paths from start to end.

    Returns:
        R_avg (3x3 list): Averaged rotation matrix after combining all paths.
        T_avg (3x1 list): Averaged translation vector after combining all paths.
    """
    R_list = []
    T_list = []

    for path in paths:
        print(f'Path: {path}')
        rotation_matrices = [data[key]['R'] for key in path]
        translation_vectors = [data[key]['T'] for key in path]
        R_final, T_final = chain_transform_matrices(rotation_matrices, translation_vectors)

        R_list.append(R_final)
        T_list.append(T_final)

    R_avg, T_avg = average_transform_matrices(R_list, T_list)

    return R_avg, T_avg


def export_main_transformations_json(input_file, output_file, main_system='m'):
    """Processes transformation data from the input JSON file to compute transformation matrices from all listed
    coordinate systems to a designated main coordinate system, and saves these transformations in a new JSON file.

    Args:
        input_file (str): Path to the input JSON file containing transformation matrices.
        output_file (str): Path where the resultant JSON file with transformations to the main system will be saved.
        main_system (str, optional): The coordinate system to which all transformations will be computed. Defaults to 'm'.

    Outputs:
        A JSON file containing transformation matrices from each available coordinate system to the main system specified.

    Example:
        If the input JSON contains transformations for 'a-b', 'b-c', and 'c-m', calling this function with the main_system
        set to 'm' will produce an output JSON with keys 'a', 'b', 'c' assuming it is possible to calculate these transformations
        based on the data provided.
    """
    with open(input_file, 'r') as file:
        data = json.load(file)

    results = {}
    # Find all starting systems, except the main system
    starts = {key.split('-')[0] for key in data.keys()}
    for start in starts:
        if start != main_system:
            # Find all paths leading to the main system
            paths = find_transformation_paths(data, start, main_system)
            # Compute transformation matrices for these paths
            R, T = compute_paths_transformation(data, paths)
            results[start] = {'R': R, 'T': T}

    # Save the computed transformations to the output file
    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)

    print(f'Transformations from all systems to the main system {main_system} saved to {output_file}')
