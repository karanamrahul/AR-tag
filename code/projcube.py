import numpy as np
import cv2
from helpers import*
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist


K = np.array([[1346.100595,0,932.1633975],
    [0,1355.933136,654.8986796],
    [0,0,1]])
world_coords_ref_tag=np.float32([[0,0],[160,0],[160,160],[0,160]])
source = 'AR-tag/results/1tagvideo.mp4'  # source the april tag video
cap = cv2.VideoCapture(source)
# Specify the path of the output video to be rendered
result = cv2.VideoWriter('result.mp4',
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        10, (1920, 1080))

#Iterating through all the frames in the Video
print('Video Rendering started ...')
Frame=0
while(cap.isOpened()):
    ret, img = cap.read()
    if ret:
        Frame+=1
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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()        
cv2.destroyAllWindows()