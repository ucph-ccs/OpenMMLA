import numpy as np


def is_tag_looking_at_another(tag1, tag2, cosine_threshold, distance_threshold):
    """Check if tag1 is looking at tag2 considering their orientation and distance."""
    if isinstance(tag1, list):
        distance = np.linalg.norm(np.array(tag1[1]).ravel() - np.array(tag2[1]).ravel())
    else:
        distance = np.linalg.norm(tag1.pose_t.ravel() - tag2.pose_t.ravel())
    if distance > distance_threshold:
        return False

    normal_vector, _ = get_outward_normal_vector(tag1)
    direction_vector = get_direction_vector(tag2, tag1)
    score = np.dot(normal_vector, direction_vector)

    return score < cosine_threshold


def get_outward_normal_vector(tag):
    """Get the outward normal vector of the tag which is perpendicular to the surface. If the normal vector is pointing
    towards the tag, flip the sign of the z-axis of the rotation matrix and the normal vector."""
    if isinstance(tag, list):
        normal = -np.dot(np.array(tag[0]), np.array([0, 0, 1])).ravel()
        tag = None
    else:
        normal = -np.dot(tag.pose_R, np.array([0, 0, 1])).ravel()
        if np.dot(normal, tag.pose_t.ravel()) > 0:
            tag.pose_R[:, 2] = -tag.pose_R[:, 2]
            normal = -normal

    return normal, tag


def get_direction_vector(start_tag, end_tag):
    """Get the normalized direction vector pointing from the start tag to the end tag."""
    if isinstance(start_tag, list):
        direction = np.array(end_tag[1]).ravel() - np.array(start_tag[1]).ravel()
    else:
        direction = end_tag.pose_t.ravel() - start_tag.pose_t.ravel()

    direction = direction / np.linalg.norm(direction)

    return direction


def is_tag_looking_at_another_2d(tag1, tag2, cosine_threshold, distance_threshold):
    """Check if tag1 is looking at tag2 considering their orientation and distance on the x-z plane."""
    if isinstance(tag1, list):
        distance = np.linalg.norm(np.array([tag1[1][0], tag1[1][2]]) - np.array([tag2[1][0], tag2[1][2]]))
    else:
        distance = np.linalg.norm(
            np.array([tag1.pose_t[0], tag1.pose_t[2]]) - np.array([tag2.pose_t[0], tag2.pose_t[2]]))

    if distance > distance_threshold:
        return False

    normal_vector, _ = get_2d_outward_normal_vector(tag1)
    direction_vector = get_2d_direction_vector(tag2, tag1)
    score = np.dot(normal_vector, direction_vector)

    return score < cosine_threshold


def get_2d_outward_normal_vector(tag):
    """Get the 2d outward normal vector of the tag which is perpendicular to the surface and project it onto the x-z
    plane. If the normal vector is pointing towards the tag, flip the sign of the z-axis of the rotation matrix
    and the normal vector."""
    if isinstance(tag, list):
        normal = -np.dot(np.array(tag[0]), np.array([0, 0, 1])).ravel()
        tag = None
    else:
        normal = -np.dot(tag.pose_R, np.array([0, 0, 1])).ravel()
        if np.dot(normal, tag.pose_t.ravel()) > 0:
            tag.pose_R[:, 2] = -tag.pose_R[:, 2]
            normal = -normal

    normal_xz = np.array([normal[0], 0, normal[2]])  # project onto x-z plane
    normal_xz /= np.linalg.norm(normal_xz)

    return normal_xz, tag


def get_2d_direction_vector(start_tag, end_tag):
    """Get the normalized direction vector pointing from the start tag to the end tag on the x-z plane."""
    if isinstance(start_tag, list):
        direction = np.array(end_tag[1]).ravel() - np.array(start_tag[1]).ravel()
    else:
        direction = end_tag.pose_t.ravel() - start_tag.pose_t.ravel()

    direction_xz = np.array([direction[0], 0, direction[2]])  # project onto x-z plane
    direction_xz = direction_xz / np.linalg.norm(direction_xz)

    return direction_xz
