from cv2 import CV_32F, rotate
import numpy as np
import cv2
from helpers import*
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist




#img = cv2.imread('/home/raul/Documents/ENPM673/Opencv/Project-1/tag_photos/tag3.jpg') # load an image
testudo_img = cv2.imread('/home/raul/Documents/ENPM673/Opencv/Project-1/testudo.png')
testudo_img = cv2.resize(testudo_img, (160, 160))
world_coords_ref_tag=np.float32([[0,0],[0,160],[160,160],[160,0]])
source = '/home/raul/Documents/ENPM673/Opencv/Project-1/1tagvideo.mp4'  # source the april tag video
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
        print("Frame:",Frame)
        for i in corner_points:
            # print((corner_points[i][0],corner_points[i][1]))
            x,y = i.ravel()
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), cv2.FILLED)
            
        matrix_ref=get_homography(corner_points,world_coords_ref_tag)
        result2 = get_warp_perspective(img, matrix_ref, (160, 160))
        tag_id,orientation=get_ar_tag_id(result2) 
        #print(int(tag_id,2),orientation)
        testudo_img=rotate_image(testudo_img,orientation)
        testudo_coords=get_world_coords(testudo_img)
        matrix = get_homography(world_coords_ref_tag,corner_points)
        warped_frame=get_warp_perspective(testudo_img,matrix, (img.shape[1],img.shape[0]))
        kernel=np.ones((3,3))
        closing= cv2.morphologyEx(warped_frame, cv2.MORPH_CLOSE, kernel)
        warped_frame = np.array(closing, dtype=np.uint8)
        corner=corner_points.copy()        
        points=np.array([[int(corner[0][0]),int(corner[0][1])],[int(corner[1][0]),int(corner[1][1])],[int(corner[2][0]),int(corner[2][1])],[int(corner[3][0]),int(corner[3][1])]])       
        blank_frame=cv2.drawContours(img,[points], -1,(0),thickness=-1)
        testudo_ar_tag=cv2.bitwise_or(warped_frame,blank_frame)
        result.write(testudo_ar_tag)
   
        draw_bound_box(testudo_ar_tag,corner_points) 
        #cv2.putText(testudo_ar_tag,'AR Tag code -'+ tag_id +'  AR-Tag ID: ' + str(int(tag_id,2)), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 120, 255), 1)
        cv2.imshow('Testudo',testudo_ar_tag)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



cap.release()        
cv2.destroyAllWindows()