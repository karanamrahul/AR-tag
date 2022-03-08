# AR-tag
AR Tag Detection, Decoding, Tracking, and Superimposing image and cube on the tag



### Authors
Rahul Karanam

### Introduction to the Project
The project was divided in three phases as explained below:<br>
1. The first task was detecting the AR-Tag corners and then using the homography concept to find the AR-Tag ID and its orientation.<br>
2. The second task was superimposing an image on the AR-Tag.<br> 
3. The third task was superimposing a 3D object on the AR-Tag.


### Software Required
To run the .py files, use Python 3. Standard Python 3 libraries like OpenCV, Numpy, Scipy and matplotlib are used.


### Finding a orientation of AR tag using Homography matrix
![](https://github.com/karanamrahul/AR-tag/blob/main/results/AR-Tag%20Decoded_screenshot_07.03.2022.png)


### Superimposing the Testudo Image onto AR Tag
![](https://github.com/karanamrahul/AR-tag/blob/main/results/Super-imposed-image_screenshot_07.03.2022.png)


### Steps to Run the code

#### Detecting AR Tag
To run the code for problem 1a, follow the following commands:

```
cd repository
python3 detect_tag.py
```
 The above code will detect the AR Tags and draw bounding box arround the tag.
 
 #### Decoding AR Tag and finding the Tag ID
To run the code for problem 1b, follow the following commands:

```
cd repository
python3 decode_tag.py
```
where the above line will output the decoded tag along with id and a example of a reference image with ID.

#### Superimposing testudo image onto the tag

To run the code for problem 2a, follow the following commands:

```
cd repository
python3 superimpose.py
```
#### Projecting a Cube onto the tag
To run the code for problem 2b, follow the following commands:
```
cd repository
python3 projcube.py
```
#### Example for Projecting Cube
![](https://github.com/karanamrahul/AR-tag/blob/main/results/projected_screenshot_07.03.2022.png)

### Video File Output Links

Problem 2a:
Output for 1tagvideo.mp4: 
https://drive.google.com/drive/folders/17uvekw6EzbJL63fmnQd51FWaF0KMO2NO?usp=sharing

Problem 2b:
Output for 1tagvideo.mp4: 
https://drive.google.com/drive/folders/17uvekw6EzbJL63fmnQd51FWaF0KMO2NO?usp=sharing


### References
The following links were helpful for this project:
1. https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
2. ENPM 673, Robotics Perception Theory behind Homography Estimation Supplementary Reference
3. https://www.learnopencv.com/tag/projection-matrix/
