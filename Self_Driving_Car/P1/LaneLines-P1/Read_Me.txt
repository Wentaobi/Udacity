The lessons quickly get you from a raw dash cam image to one with a collection of lines overlaid on the right and left edges of the lane ahead, even if the lane markings are broken! I had just happened to learn about these techniques the previous week in the several lessons I had watched in the Udacity Intro to Computer Vision course, luckily enough. It’s fairly easy to understand, though. The steps were:

Convert RGB to grayscale
Apply a slight Gaussian blur
Perform Canny edge detection
Define a region of interest and mask away the undesired portions of the image
Retrieve Hough lines
Apply lines to the original image
All of this uses the OpenCV computer vision and image processing library, which makes the process incredibly simple. The only difficulty is in fiddling with the parameters of each function to produce the desired output. The Gaussian blur takes a kernel size - the bigger the blurrier. The Canny edge takes a high and low threshold - which determine a minimum difference in intensity to establish an edge and to form a contiguous extension of an established edge, respectively. The Hough transform takes a resolution for line position and orientation, a minimum number of points to establish a line, the minimum length of a line, and the maximum gap between points allowed for a line. Of course, tuning the Hough parameters proved to be the most time consuming since it has by far the most parameters.

This pipeline worked well enough, and it was possible to get lines (typically a small collection of lines) running along both the left and right edges of the lane. The next step for the project was to consolidate these collections of lines into a single line for each side of the lane, and to extrapolate that line to the bottom edge of the image and to a point of our choosing near the center of the image. This effectively kept the lines at a fairly constant length, and precluded having gaps between the first broken line and the bottom of the image.

At this point, the pipeline is applied to two videos and if it performs well enough the project can be considered complete, but Udacity tossed in a bonus challenge which, of course, I couldn’t resist. The final video includes a more curvy patch of road, areas of shadows being cast onto the lane and other bits of discoloration, and (grrrr!) a bridge that is considerably lighter-colored than the rest of the road surface.

The bridge was especially difficult because there wasn’t enough contrast between the solid yellow line on the left and the road surface to trigger a Canny edge using the pipeline as is. It’s possible that fiddling with the existing parameters might have solved the problem, but someone in the nanodegree forum suggested converting to HSV (hue, saturation, value) colorspace to boost the yellow areas of the image and I decided to try it out. In the end my pipeline looked like this:

Convert RGB to grayscale
Darken the grayscale (to reduce contrast in discolored sections of road)
Convert RGB to HSV
Isolate yellow in HSV to produce a yellow mask
Also isolate white in HSV to produce a white mask
Combine yellow and white masks with bitwise OR
Apply combined mask to darkened grayscale with bitwise OR
Apply a slight Gaussian blur
Perform Canny edge detection
Define a region of interest and mask away the undesired portions of the image
Retrieve Hough lines
Consolidate and extrapolate lines and apply them to the original image
