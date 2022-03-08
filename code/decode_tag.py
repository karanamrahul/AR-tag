
# 1-b Perception
## Decoding the Tag to get the tag-id and orientation
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from helpers import*



img=cv2.imread('/home/raul/Documents/ENPM673/Opencv/Project-1/tag_photos/tag54.jpg')
img_tag_ref=cv2.imread('/home/raul/Documents/ENPM673/Opencv/Project-1/marker_tag.jpeg')  # source the april tag reference image
world_coords=np.float32([[0,0],[160,0],[160,160],[0,160]])
image_fft=fft_ifft(img)
corner_points=get_corners(image_fft)
matrix = get_homography(corner_points, world_coords)
result = get_warp_perspective(img, matrix, (160, 160))
tag_id,orientation=get_ar_tag_id(result) 
tag_id_2,_=get_ar_tag_id(img_tag_ref)
draw_bound_box(img,corner_points) 
print('AR-Tag ID1: ' + str(int(str(tag_id),2)))
cv2.putText(img,'AR Tag code -'+ tag_id +'  AR-Tag ID: ' + str(int(tag_id,2)), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
print('AR-Tag ID2: ' + str(int(str(tag_id_2),2)))
cv2.putText(img_tag_ref,'AR Tag code -'+ tag_id_2 +'  AR-Tag ID: ' + str(int(tag_id_2,2)), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
cv2.imshow("Image AR Tag detection",result) # This is the tag detected from the image frame.
cv2.imshow('AR-Tag Decoded', img) # Show final decoded AR Tag with ID.
cv2.imshow("Reference Tag",img_tag_ref)
cv2.waitKey(0)
cv2.destroyAllWindows()