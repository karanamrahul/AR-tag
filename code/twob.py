from cv2
import numpy as np
import cv2
from helpers import*
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist

""" Projection Matrix

Intrinsic Matrix Parameters of the camera - K
Projection Matrix Parameters of the camera - P
Rotation Matrix Parameters - R
Translation Vector Parameters - T

P = K[ R | T ]

B~=K^(-1)*H


1.  First we find the homography matrix using the corresponding points from the corners and the cube dimensions
2.  We find the B_hat by taking the dot product of the Homography matrix and the inverse of Intrinsic matrix (K). 
3.  After computing B_hat , we then find the lambda (λ) - (2/norm(b1)+norm(b2)) B=[b1 b2 b3] scale factor which is the 
    first two column vectors of the B_hat
4.  B_hat=[λ*b1 λ*b2 λ*b3] which is equivalent to [r1,r2,t]- rotational vectors and translation vectors.
5.  To get r3 we take cross product of r1 and r2.
6.  Finally, we take a dot product of the camera_matrix(K) and [R t] to get our projection matrix.
"""

def get_projection_matrix(homo_mat,intrinsic_mat):
    intrinsic_mat_inv = np.linalg.inv(intrinsic_mat)
    B_mat = intrinsic_mat_inv.dot(homo_mat)

    # Now we check whether the norm is positive or not as we get two possible solutions(infront and behind the camera)
    if np.linalg.norm(B_mat)>0:
        B=1*B_mat
    else:
        B=-1*B_mat

    # Lambda Scaling factor ((λ)) - the first two columns of B matrix B=[b1 b2 b3]. 
    lambda_scale=2/(np.linalg.norm(np.matmul(intrinsic_mat_inv, homo_mat[:,0])) + np.linalg.norm(np.matmul(intrinsic_mat_inv, homo_mat[:,1])))

    # We find the rotation matrix (R) and translation vectors (T) 
    # B=[λ*b1 λ*b2 λ*b3]
    r_1 = lambda_scale*B[:,0]
    r_2 = lambda_scale*B[:,1]
    r_3 = np.cross(r_1, r_2)/lambda_scale
    t = np.array([lambda_scale*B[:,2]]).T
    R = np.array([r_1,r_2,r_3]).T
    R = np.hstack([R, t])
    
    # P = K * [R | t]
    proj_mat = np.matmul(intrinsic_mat, np.matrix(R))

    return proj_mat

def get_project_points(image,projection_matrix):
    cube_dim=160
    xyz_coords = [[0, 0, 0, 1],
    [0, cube_dim, 0, 1],
    [cube_dim, cube_dim, 0, 1],
    [cube_dim, 0, 0, 1],
    [0, 0, -cube_dim, 1],
    [0, cube_dim, -cube_dim, 1],
    [cube_dim, cube_dim, -cube_dim, 1],
    [cube_dim, 0, -cube_dim, 1]]

    projection_matrix=np.array(projection_matrix)
    new_cube = np.array([projection_matrix.dot(cube_itr) for cube_itr in xyz_coords])

    points = new_cube[:, :-1] / new_cube[:, 2:]


    # image = cv2.line(image, tuple(points[0].astype(int)), tuple(points[1].astype(int)), (0, 0, 255), 2)
    # image = cv2.line(image, tuple(points[1].astype(int)), tuple(points[2].astype(int)), (0, 0, 255), 2)
    # image = cv2.line(image, tuple(points[2].astype(int)), tuple(points[3].astype(int)), (0, 0, 255), 2)
    # image = cv2.line(image, tuple(points[3].astype(int)), tuple(points[0].astype(int)), (0, 0, 255), 2)
    # image = cv2.line(image, tuple(points[4].astype(int)), tuple(points[5].astype(int)), (0, 0, 255), 2)
    # image = cv2.line(image, tuple(points[5].astype(int)), tuple(points[6].astype(int)), (0, 0, 255), 2)
    # image = cv2.line(image, tuple(points[6].astype(int)), tuple(points[7].astype(int)), (0, 0, 255), 2)
    # image = cv2.line(image, tuple(points[7].astype(int)), tuple(points[4].astype(int)), (0, 0, 255), 2)
    # image = cv2.line(image, tuple(points[0].astype(int)), tuple(points[4].astype(int)), (0, 0, 255), 2)
    # image = cv2.line(image, tuple(points[1].astype(int)), tuple(points[5].astype(int)), (0, 0, 255), 2)
    # image = cv2.line(image, tuple(points[2].astype(int)), tuple(points[6].astype(int)), (0, 0, 255), 2)
    # image = cv2.line(image, tuple(points[3].astype(int)), tuple(points[7].astype(int)), (0, 0, 255), 2)

    # for pts in points:
    #     cv2.circle(image, (int(pts[0]), int(pts[1])), 3, (180, 0, 0), -1)
        
    return points
def points(dst_cube,img_dst_cp):

   

    img_dst_cp = cv2.line(img_dst_cp, tuple(dst_cube[0].astype(int)), tuple(dst_cube[1].astype(int)), (0, 0, 255), 2)
    img_dst_cp = cv2.line(img_dst_cp, tuple(dst_cube[1].astype(int)), tuple(dst_cube[2].astype(int)), (0, 0, 255), 2)
    img_dst_cp = cv2.line(img_dst_cp, tuple(dst_cube[2].astype(int)), tuple(dst_cube[3].astype(int)), (0, 0, 255), 2)
    img_dst_cp = cv2.line(img_dst_cp, tuple(dst_cube[3].astype(int)), tuple(dst_cube[0].astype(int)), (0, 0, 255), 2)
    img_dst_cp = cv2.line(img_dst_cp, tuple(dst_cube[4].astype(int)), tuple(dst_cube[5].astype(int)), (0, 0, 255), 2)
    img_dst_cp = cv2.line(img_dst_cp, tuple(dst_cube[5].astype(int)), tuple(dst_cube[6].astype(int)), (0, 0, 255), 2)
    img_dst_cp = cv2.line(img_dst_cp, tuple(dst_cube[6].astype(int)), tuple(dst_cube[7].astype(int)), (0, 0, 255), 2)
    img_dst_cp = cv2.line(img_dst_cp, tuple(dst_cube[7].astype(int)), tuple(dst_cube[4].astype(int)), (0, 0, 255), 2)
    img_dst_cp = cv2.line(img_dst_cp, tuple(dst_cube[0].astype(int)), tuple(dst_cube[4].astype(int)), (0, 0, 255), 2)
    img_dst_cp = cv2.line(img_dst_cp, tuple(dst_cube[1].astype(int)), tuple(dst_cube[5].astype(int)), (0, 0, 255), 2)
    img_dst_cp = cv2.line(img_dst_cp, tuple(dst_cube[2].astype(int)), tuple(dst_cube[6].astype(int)), (0, 0, 255), 2)
    img_dst_cp = cv2.line(img_dst_cp, tuple(dst_cube[3].astype(int)), tuple(dst_cube[7].astype(int)), (0, 0, 255), 2)

    for pts in dst_cube:
        cv2.circle(img_dst_cp, (int(pts[0]), int(pts[1])), 3, (180, 0, 0), -1)
        
    

def draw_cube(image,points):
    # The base
    image = cv2.drawContours(image, [points[:4]], -1, (0, 255, 0), 3)

    # Drawing lines or pillars 
    for i, j in zip(range(4), range(4, 8)):
        image = cv2.line(image, tuple(points[i].astype(int)), tuple(points[j].astype(int)), (255, 255, 0), 3)

    # Creating the top part for the cube
        image = cv2.drawContours(image, [points[4:]], -1, (0, 0, 255), 3)
    return image


img = cv2.imread('/home/raul/Documents/ENPM673/Opencv/Project-1/tag_video.jpeg')
K = np.array([[1346.100595,0,932.1633975],
    [0,1355.933136,654.8986796],
    [0,0,1]])
dim=161
#xyz_coords = np.float32([[0, 0, 0], [0, 160, 0], [160, 160, 0], [160, 0, 0], [0, 0, -160], 
#                               [0, 160, -160], [160, 160, -160], [160, 0, -160]])

world_coords_ref_tag=np.float32([[0,0],[160,0],[160,160],[0,160]])

image_fft = fft_ifft(img)
corner_points = get_corners(image_fft)
print(corner_points)
print("ref",world_coords_ref_tag)
for i in range(0, 4):
    cv2.circle(img, (int(corner_points[i][0]), int(corner_points[i][1])), 5, (0, 0, 255), cv2.FILLED)
    
matrix_ref=get_homography(world_coords_ref_tag,corner_points)
proj_mat=get_projection_matrix(matrix_ref,K)
final_points=get_project_points(img,proj_mat)
final_points = np.int32(final_points).reshape(-1,2)
img = cv2.drawContours(img, [final_points[:4]],-1,(0,255,255),3)
for i,j in zip(range(4),range(4,8)):
    img = cv2.line(img, tuple(final_points[i]), tuple(final_points[j]),(0,0,255),3)
    #Drawing Top layer in Red 
    img = cv2.drawContours(img, [final_points[4:]], -1, (0,255,200), 3)
# print("Proj",proj_mat)
img_c=img.copy()
# cube=draw_cube(img,points)

cube = points(final_points,img_c)
#cv2.putText(frame_empty,'AR Tag code -'+ tag_id +'  AR-Tag ID: ' + str(int(tag_id,2)), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 120, 255), 1)
cv2.imshow('image after fft',img_c)
cv2.imshow('Testudoh',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
