import numpy as np
import cv2

def get_rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def angle(v1, v2):
    """ get angle between two vectors. """
    ang = np.arccos(v1@v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return -1 * np.sign(np.cross(v1, v2)) * ang

def subdivide_linestring(points: np.ndarray, gap=20):
    cumsum_dist = np.cumsum(np.linalg.norm(points[:-1] - points[1:], axis=1))
    x = np.linspace(0, cumsum_dist[-1], int(cumsum_dist[-1] / gap))
    samples = np.vstack([x, np.zeros_like(x)]).T

    d = [0] + cumsum_dist.tolist()
    xy = []
    for i in range(len(points)-1):
        p0, p1 = points[[i, i+1]]
        # can be done more efficiently
        m = np.bitwise_and(d[i]-np.finfo(float).eps < x, x < d[i+1]) 
        R = get_rotation_matrix(angle(p1-p0, [1, 0]))
        xy.append(p0 + (samples[m] - [d[i], 0]) @ R.T)
    return np.vstack(xy)


def plot(img, points: np.ndarray, color, thickness, linestyle="-", gap=20, alpha=1.):
    im = img.copy()
    if linestyle not in ("-", "-.", "--", ":"):
        raise NotImplementedError(f"Linestyle: {linestyle} not implemented.")
        
    xy = points if linestyle == "-" else subdivide_linestring(points, gap)
    xy = np.rint(xy).astype(int)
    
    if linestyle == "-":
        for p1, p2 in zip(xy[:-1], xy[1:]):
            cv2.line(im, p1, p2, color, thickness)
    elif linestyle == "--":
        for i, (p1, p2) in enumerate(zip(xy[:-1], xy[1:])):
            if (i+1) % 3:
                cv2.line(im, p1, p2, color, thickness)
    elif linestyle == ":":
        for p in xy:
            cv2.circle(im, p, thickness, color, -1)
    elif linestyle == "-.":
        for i in range(len(xy)-1):
            if (i%4) == 0: 
                cv2.line(im, *xy[[i, i+1]], color, thickness)
            if (i%4) == 2:
                p = np.rint(xy[i] + (xy[i+1]-xy[i])/2).astype(int) # interpolation
                cv2.circle(im, p, thickness, color, -1)
    return cv2.addWeighted(im, alpha, img, 1 - alpha, 0, im)