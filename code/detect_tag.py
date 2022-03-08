
# 1-a Perception
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist

from helpers import*


#img = cv2.imread('/home/raul/Documents/ENPM673/Opencv/Project-1/tag_photos/tag3.jpg') # load an image

source = 'AR-tag/results/1tagvideo.mp4'  # source the april tag video
world_coords=np.float32([[0,0],[160,0],[160,160],[0,160]])
cap = cv2.VideoCapture(source)
show_plot=True  
# Uncomment this line to get the plot showing the FFT,FFT+mask and edges.
while(cap.isOpened()):
    ret, img = cap.read()
    if ret:
        image_fft=fft_ifft(img,show_plot)
        corner_points=get_corners(image_fft)
        matrix = get_homography(corner_points, world_coords)
        result = get_warp_perspective(img, matrix, (160, 160))
        tag_id,orientation=get_ar_tag_id(result) 
        draw_bound_box(img,corner_points) 
        print('AR-Tag ID: ' + str(int(str(tag_id),2)))
        cv2.putText(img, 'AR-Tag ID: ' + str(tag_id), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
        
        cv2.imshow('AR-Tag Decoded', img) # Show final decoded AR Tag with ID.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

    





