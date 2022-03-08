import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from sympy import Matrix



def get_ar_tag_id(image):
    """_summary_
        This function will convert the image to 8 x 8 grid (16 cells) 
        which will be used to find the orientation of the tag.
        
        We check the orientation of the tag with the reference tag orientation and decoding scheme.
        
        Please find the marker_tag reference tag in the images/marker_tag.png
    Args:
        image (numpy nd array): _This is the input image after warping using the homography matrix_

    Returns:
        _binary_id_tag and the tag id _: 
    """
    image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape
    x_grid = int(w/8)
    y_grid = int(h/8)

    cells = []

    for y in range(2 * y_grid, h - (2 * y_grid), y_grid): 

        for x in range(2 * x_grid, w - (2 * x_grid), x_grid):
            block = image[y : y + y_grid, x : x + x_grid]
            cells.append(block)

            # Checks the values of the outer corners. If 1 then ID the tag starting from the inner opposing corner proceeding clockwise.
    idx = [0, 3, 5, 6, 9, 10, 12, 15]
    id = []
    for i in range(len(idx)):
        if np.mean(cells[idx[i]]) > 127:
            id.append(1)
        else:
            id.append(0)

    if id[idx.index(0)] == 1: # Check upper left corner
        # Start at [10], [9], [5], [6]
        tag_binary_id = str(str(id[idx.index(10)]) + str(id[idx.index(9)]) + str(id[idx.index(5)]) + str(id[idx.index(6)]))[::-1]
        orientation = 1 # Should rotate 180 clockwise
        return tag_binary_id,orientation
    elif id[idx.index(3)] == 1: # Check upper right corner
        # Start at [9], [5], [6], [10]
        tag_binary_id = str(str(id[idx.index(9)]) + str(id[idx.index(5)]) + str(id[idx.index(6)]) + str(id[idx.index(10)]))[::-1]
        orientation=2 # Should rotate 90 clockwise
        return tag_binary_id,orientation
    elif id[idx.index(12)] == 1: # Check lower left corner
        # Start at [6], [10], [9], [5]
     
        tag_binary_id = str(str(id[idx.index(6)]) + str(id[idx.index(10)]) + str(id[idx.index(9)]) + str(id[idx.index(5)]))[::-1]
        orientation=3
        return tag_binary_id,orientation
    elif id[idx.index(15)] == 1: # Check lower right corner
        # Start at [5], [6], [10], [9]
        tag_binary_id = str(str(id[idx.index(5)]) + str(id[idx.index(6)]) + str(id[idx.index(10)]) + str(id[idx.index(9)]))[::-1]
        orientation=0 # Perfect orientation w.r.t the ref tag
        return tag_binary_id,orientation
        
def rotate_image(image,orientation):
    if orientation == 0:
        return image
    elif orientation == 1:
        return cv2.rotate(image,cv2.ROTATE_180)
    elif orientation == 2:
        return cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 3:
        return cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
    
def get_world_coords(image):
    h,w,_=image.shape
    return np.float32([[0,0],[h,0],[h,w],[0,h]])


def fft_ifft(img,show_plot=False):
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_gray,(7,7),cv2.BORDER_DEFAULT)
    opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, np.ones((7,7)))
    r,image_fft=cv2.threshold(opening,170,255,cv2.THRESH_BINARY)
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    img=clahe.apply(image_fft)
    
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft) 
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    mask = np.ones((rows, cols, 2), np.uint8)
    r = 240
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0


    # apply mask and inverse DFT
    fshift = dft_shift * mask

    fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    #ret, img_back = cv2.threshold(img_back,170,255,cv2.THRESH_BINARY)
    img_back = cv2.GaussianBlur(img_back,(7,7),cv2.BORDER_DEFAULT)

    
  
    if show_plot==True:
        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2,2,1)
        ax1.imshow(img, cmap='gray')
        ax1.title.set_text('Input Image')
        ax2 = fig.add_subplot(2,2,2)
        ax2.imshow(magnitude_spectrum, cmap='gray')
        ax2.title.set_text('FFT of image')
        ax3 = fig.add_subplot(2,2,3)
        ax3.imshow(fshift_mask_mag, cmap='gray')
        ax3.title.set_text('FFT + Mask')
        ax4 = fig.add_subplot(2,2,4)
        ax4.imshow(img_back, cmap='gray')
        ax4.title.set_text('After inverse FFT - Edges')
        plt.show()
    return img_back

# reference : https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def order_points(points):
    # sort points along columns
    col_sorted_points = points[np.argsort(points[:, 0]), :]
    
    # left_col_points have top left and bottom left and right_col_points have top right and bottom right
    left_col_points = col_sorted_points[:2, :]
    right_col_points = col_sorted_points[2:, :]
    
    # sort leftmost according to rows
    left_col_points = left_col_points[np.argsort(left_col_points[:, 1]), :]
    (tl, bl) = left_col_points
    
    # now get top right and bottom right
    euc_dist = dist.cdist(tl[np.newaxis], right_col_points, "euclidean")[0]
    (br, tr) = right_col_points[np.argsort(euc_dist)[::-1], :]
    return np.array([tl, tr, br, bl], dtype="float32")



def draw_bound_box(img,corner_points):
# Draw the bounding boxes
    cv2.line(img, (int(corner_points[3][0]),int(corner_points[3][1])),(int(corner_points[0][0]),int(corner_points[0][1])), (125, 255, 0), 6)
    for i in range(0,3):
        cv2.line(img, (int(corner_points[i][0]),int(corner_points[i][1])),(int(corner_points[i+1][0]),int(corner_points[i+1][1])), (125, 255, 0), 6)
    for i in range(0,4):
    #print((corner_points[i][0],corner_points[i][1]))
     cv2.circle(img,(int(corner_points[i][0]),int(corner_points[i][1])),5,(0,0,255),cv2.FILLED)
     
def is_on_right_side(x, y, xy0, xy1):
    x0, y0 = xy0
    x1, y1 = xy1
    a = float(y1 - y0)
    b = float(x0 - x1)
    c = - a*x0 - b*y0
    return a*x + b*y + c > 10000

def test_point(x, y, vertices):
    num_vert = len(vertices)
    is_right = [is_on_right_side(x, y, vertices[i], vertices[(i + 1) % num_vert]) for i in range(num_vert)]
    all_left = not any(is_right)
    all_right = all(is_right)
    return all_left or all_right 

def get_corners(img):
    
    corners = cv2.goodFeaturesToTrack(img, 20 , 0.05 , 50)
    corners = np.int0(corners)
    x_min=np.argmin(corners[:,:,0])
    x_max=np.argmax(corners[:,:,0])
    y_min=np.argmin(corners[:,:,1])
    y_max=np.argmax(corners[:,:,1])


    corner_tag=[]
    top_y=np.float('inf')
    left_x=np.float('inf')
    points=np.float32([[corners[x_min][0][0],corners[x_min][0][1]],[corners[y_min][0][0],corners[y_min][0][1]],[corners[x_max][0][0],corners[x_max][0][1]],[corners[y_max][0][0],corners[y_max][0][1]]])
    for corners in corners:
            x,y = corners.ravel()
            if test_point(x, y, points):
                corner_tag.append([x,y])
                if top_y > y:
                    top_y = y
                    top_x = x
                if left_x > x:
                    left_x = x
                    left_y = y
  
    #print(corner)
    
    corner_tag_2=[]
    corner_tag=np.asarray(corner_tag)
    x_min_=np.argmin(corner_tag[:,0])
    x_max_=np.argmax(corner_tag[:,0])
    y_min_=np.argmin(corner_tag[:,1])
    y_max_=np.argmax(corner_tag[:,1])
    corner_tag_new=np.array([corner_tag[x_min_], corner_tag[y_min_],corner_tag[x_max_],corner_tag[y_max_]])
    # points=np.float32([[corners[x_min][0],corners[x_min][1]],[corners[y_min][0][0],corners[y_min][0][1]],[corners[x_max][0][0],corners[x_max][0][1]],[corners[y_max][0][0],corners[y_max][0][1]]])
    index=[x_min_,x_max_,y_min_,y_max_]
    corner_tag=np.delete(corner_tag,index,0)
    for corners in corner_tag:
            x,y = corners.ravel()
            if test_point(x, y, corner_tag_new):
                corner_tag_2.append([x,y])
                if top_y > y:
                    top_y = y
                    top_x = x
                if left_x > x:
                    left_x = x
                    left_y = y
                    
    corner_tag_2=np.asarray(corner_tag_2)
    x_min_=np.argmin(corner_tag_2[:,0])
    x_max_=np.argmax(corner_tag_2[:,0])
    y_min_=np.argmin(corner_tag_2[:,1])
    y_max_=np.argmax(corner_tag_2[:,1])
    corner_tag_new_2=np.array([corner_tag_2[x_min_], corner_tag_2[y_min_],corner_tag_2[x_max_],corner_tag_2[y_max_]])

    # Corners for the april tag- each frame
    corner_points=order_points(corner_tag_new_2)
    #print(corner_points.shape)
    return corner_points

def get_warp_perspective(image,homography,target):
    """_summary_
    This function will return the final warped image given the given homography matrix and target dimension.
    
    src = H * dst 
    Apply the Warping on each row and column of the source image by adding one to it. 
    src - [x,y,1] dst - [x,y,z] 

    By verifying with the boundary limit of the dimension of the target image, we 
    copy the value to the target image from the source image. This
    Args:
        image (_type_): The source image
        homography (_type_): Matrix corresponding to the homography between the corners of the tag and world coords.
        target (_type_): The target image to which we are warping.
    """

    image = cv2.transpose(image)
    trg = np.zeros((target[0],target[1],3))
    h,w,_ = image.shape
    for row in range(0,h):
        for col in range(0,w):
            dst_coords = homography.dot([row,col,1])
            dst_row,dst_col,_=(dst_coords / dst_coords[2]).astype(int)
            if (dst_row > 0 and dst_row < target[0]) and (dst_col > 0 and dst_col < target[1]) :
                trg[dst_row,dst_col]=image[row,col]
    warped_image = np.array(trg, dtype=np.uint8)
    warped_image = cv2.transpose(warped_image) 
    return warped_image
   
    

# Find the Homography Matrix for the given two plane object and point correspondences.

# Corresponding points of the two planes

# homography : This method is used to calculate the least square solution or find the solution for a homogenous system of equation.


def get_homography(corner_points,world_coords):
    x1 = corner_points[0][0]
    x2 = corner_points[1][0]
    x3 = corner_points[2][0]
    x4 = corner_points[3][0]
    x_1 = world_coords[0][0]
    x_2 = world_coords[1][0]
    x_3 = world_coords[2][0]
    x_4 = world_coords[3][0]
    y1 = corner_points[0][1]
    y2 = corner_points[1][1]
    y3 = corner_points[2][1]
    y4 = corner_points[3][1]
    y_1 = world_coords[0][1]
    y_2 = world_coords[1][1]
    y_3 = world_coords[2][1]
    y_4 = world_coords[3][1]

    # The Matrix which we use to find the solution to the homogenous system of equations AX = 0.
    mat = np.matrix([[-x1, -y1,  -1, 0, 0, 0, x1*x_1, y1*x_1, x_1],
                [0, 0, 0, -x1, -y1, -1, x1*y_1, y1*y_1, y_1],
                [-x2, -y2, -1, 0, 0, 0, x2*x_2, y2*x_2, x_2],
                [0, 0, 0, -x2, -y2, -1, x2*y_2, y2*y_2, y_2],
                [-x3, -y3, -1, 0, 0, 0, x3*x_3, y3*x_3, x_3],
                [0, 0, 0, -x3, -y3, -1, x3*y_3, y3*y_3, y_3],
                [-x4, -y4, -1, 0, 0, 0, x4*x_4, y4*x_4, x_4],
                [0, 0, 0, -x4, -y4, -1, x4*y_4, y4*y_4, y_4]])


    # Singular Value Decomposition  SVD(A)= U*S*V
    # Dimensions of A: m x n , U: m x m , S: m x n , V: n x n

    u = mat*mat.T
    v = mat.T*mat
    val_u,vec_u=np.linalg.eig(u)
    val_v,vec_v=np.linalg.eig(v)
    # Sorting eigen vectors based upon the decreasing order of eigen values
    idx = val_v.argsort()[::-1]   
    val_v = val_v[idx]
    vec_v = vec_v[:,idx]

    idx = val_u.argsort()[::-1]   
    val_u = val_u[idx]
    vec_u = vec_u[:,idx]

    if mat.shape[0] > mat.shape[1]:
     s_val=np.mat(np.sqrt(np.diag(val_u)))
     s_val=s_val[:,:mat.shape[1]]
    else:
        s_val=np.mat(np.sqrt(np.diag(val_v)))
        s_val=s_val[:mat.shape[0],:]

    homo_mat = np.zeros((mat.shape[1], 1))
    for index in range(0, mat.shape[1]):
        homo_mat[index, 0] = vec_v[index, vec_v.shape[1] - 1]
    homo_mat=homo_mat.reshape((3,3))   
    for index1 in range(0, 3):
        for index2 in range(0, 3):
            homo_mat[index1][index2] = homo_mat[index1][index2] / homo_mat[2][2]
    return homo_mat

    

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
