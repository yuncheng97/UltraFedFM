import cv2
import numpy as np


def find_tangent_lines(ellipse, reference_point):
    """Find the Ellipse's two tangents that go through a reference point

    Args:
        ellipse:
          center: The center of the ellipse
          axes: Major and Minor axes of the ellipse
          rotation: The counter-clockwise rotation of the ellipse in radians
        reference_point: The coordinates of the reference point.

    Return:
        (m1, h1): Slope and intercept of the first tangent.
        (m2, h2): Slope and intercept of the second tangent.
    """
    (x0, y0), axes, rotation  = ellipse
    a, b = axes[0]/2, axes[1]/2
    rotation =  np.radians(rotation)
    s, c = np.sin(rotation), np.cos(rotation)
    p0, q0 = reference_point

    A = (-a ** 2 * s ** 2 - b ** 2 * c ** 2 + (y0 - q0) ** 2)
    B = 2 * (c * s * (a ** 2 - b ** 2) - (x0 - p0) * (y0 - q0))
    C = (-a ** 2 * c ** 2 - b ** 2 * s ** 2 + (x0 - p0) ** 2)

    if B ** 2 - 4 * A * C < 0:
        raise ValueError('Reference point lies inside the ellipse')

    t1, t2 = (
        (-B + np.sqrt(B ** 2 - 4 * A * C)) / (2 * A),
        (-B - np.sqrt(B ** 2 - 4 * A * C)) / (2 * A),
    )
    return (
        (1 / t1, q0 - p0 / t1),
        (1 / t2, q0 - p0 / t2),
    )


# Coordinates of the ellipse's major axes
def get_major_axis(ellipse):

    center, axes, angle = ellipse
    major_axis_angle_rad = np.radians(angle)
    major_axis_length = max(axes)
    cos_angle = np.cos(major_axis_angle_rad)
    sin_angle = np.sin(major_axis_angle_rad)

    major_axis_endpoint1 = (
        int(center[0] - 0.5 * major_axis_length * sin_angle),
        int(center[1] + 0.5 * major_axis_length * cos_angle)
    )
    major_axis_endpoint2 = (
        int(center[0] + 0.5 * major_axis_length * sin_angle),
        int(center[1] - 0.5 * major_axis_length * cos_angle)
    )

    return major_axis_endpoint1, major_axis_endpoint2

# Angle beteen 2 lines
def calculate_angle(lineA, lineB):

    vA = np.array([(lineA[0][0] - lineA[1][0]), (lineA[0][1] - lineA[1][1])])
    vB = np.array([(lineB[0][0] - lineB[1][0]), (lineB[0][1] - lineB[1][1])])

    angle = np.arccos(np.dot(vA, vB) / np.dot(vA, vA) ** 0.5 / np.dot(vB, vB) ** 0.5)
    deg = np.rad2deg(angle) 

    if deg - 180 >= 0:
        return 360 - ang_deg
    else:
        return deg
        
        
def angle_of_progression_estimation(label, return_img=False):        
    tmp_img = np.zeros_like(label).astype(np.uint8)
    # PS ellipse
    contours_ps, _ = cv2.findContours((label == 1).astype(np.uint8), cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    largest_contour_ps = max(contours_ps, key=cv2.contourArea)
    ellipse_ps = cv2.fitEllipse(largest_contour_ps)
    
    # PS ellipse: major axis points
    (x1, y1), (x2, y2) = get_major_axis(ellipse_ps)

    ps_points = [(x1, y1), (x2, y2)] if x1 > x2 else [(x2, y2), (x1, y1)]
    ps_pont = (int(ps_points[0][0]), int(ps_points[0][1]))
    
    # FH ellipse
    contours_fh, _ = cv2.findContours((label == 2).astype(np.uint8), cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)
    largest_contour_fh = max(contours_fh, key=cv2.contourArea)
    ellipse_fh = cv2.fitEllipse(largest_contour_fh)

    # Find tangent line
    (m1, h1), (m2, h2) = find_tangent_lines(ellipse=ellipse_fh, reference_point=ps_pont)

    tmp_p_disp = 150
    op = ((ps_points[0][1] + tmp_p_disp - h1) / m1)
    
    # Draw if needed
    if return_img:
        cv2.ellipse(tmp_img, ellipse_ps, (120, 255, 255), 1, cv2.LINE_AA) 
        cv2.line(tmp_img, (int(x1), int(y1)), (int(x2), int(y2)), (160, 0, 255), 2) 
        cv2.ellipse(tmp_img, ellipse_fh, (160, 255, 255), 1, cv2.LINE_AA)
        cv2.line(tmp_img, ps_pont, (int(op), int(ps_points[0][1] + tmp_p_disp)), (240, 0, 0), 2)
    
    # Angle estimation
    aop = calculate_angle([ps_points[0], ps_points[1]], [ps_pont, (op, ps_points[0][1] + tmp_p_disp)])
    
    if return_img:
        return aop, tmp_img
    else:
        return aop