function [P,Pcrop,imgs,Charts,Scale_extracted,varargout] = corealign(folder,varargin)  

[P,Pcrop,imgs,Charts,varargout] = corealign(folder,varargin)  

This function aligns a sequence of overlapping images of sediment cores 
using extracted Speeded Up Robust Feature (SURF) points. Features are matched 
between two adjacent images. The translation vector between the matched points 
are used to calculate the location in a panorama for each image. If an object 
such as sample label or color chart is ridigly attached to the camera such that 
it *should* be visible in the same location in each image, then this object 
can be used for tilt correction caused by e.g., camera vibration, which can be 
caused by sliding the camera on rails or ship movement. If this object includes 
a color chart any variable intensity from light flickering can also be corrected.  

REQUIRED INPUT VARIABLES 
================================ 
'folder' | string containing folder name with image sequence (uint8 or uint16).  

OUTPUT VARIABLES 
================================ 
'P' | Panorama of aligned images as either uint8 or uint16 depending on input images. 
'imgs' | cell array of all images after color adjustment. 
'Charts' | cell arry of all color charts extracted from images after color adjustment.  

OPTIONAL ARGUMENT / VALUE PAIRS 
================================  

'alt' | [1,0] Default: 0 
Alternate translation method is performed if 1. If tilt correction is performed, 
a new set of SURF points are detected and features matched for each pair of images. 
Translation distances are calculated from the new matched pairs of SURF points 
Default method: SURF points are inversely transformed using tilt correction matrix 
Alternate method: After tilt correction of images, a new set of SURF points are obtained  

'color' | [1,0] Default: 1 
Adjusts color values of all images to match a reference image in the sequence by detecting 
and extracting the color chart from every image. The mean RGB values are calculated for each 
Chart, and the chart with values closest to the median of all becomes the reference. 
This removes variable intensity between images that could be caused by lighting flicker. 
Images of the color charts are stored in /private. Additional images of charts can be added. 
Samples of color charts are contined in /private. Additional images of charts can bee added.  

'tilt' | [1,0] Default: 1 
Corrects for variable tilt between two image pairs that could be caused by camera movement 
by using the translation vectors of SURF points corresponding to a static object that 
*should* be in the same location in both image pairs. Examples of static objects are 
sample labels and color charts that move rigidly with the camera. 
Manually moved objects should not be used!  

'fig' | [1,0] Default: 0 
display a figure with progress of alignment. Looks cool I guess but adds time  

'crop' | [1,0] Default: 0 
Experimental automatic cropping of the image and determination of core length. 
The first (left-most) and last (right-most) images are compared to image templates. 
The first image is compared to a the first 5-cm of the scale The last image is compared 
an image of 3D-printed block which must be placed against the bottom of the core. 
If these features are detected correctly, pixel dimensions of the core are obtained 
allowing it to be segmented from the background. The top and bottom pixel locations of 
the scale bar are also obtained. Next, the scale bar is segmented from the core image 
and compared to a template scale image of a known length and pixel number. 
A geometric transformation is obtained to warp the extracted scalebar to the template. 
Black fill value are used, and the number of non-black pixels in the warped image is used 
to calculate length in centimeters based on the number of pixels / cm in the template image. 
REQUIREMENTS FOR AUTOCROP AND LENGTH DETERMINATION 
The scale and bottom block used when photographing cores must be visually identical to the templates 
The scale template must be as long or longer than the core. 
NOTE: the input parameter 'MaximumDistance' needs to be specified or the geometric transformation  
will likely fail because individual tick marks all contain similar features. Small changes to the 
maximum distance yield slightly different results. 
VARARGOUT returns an 1x4 array with: 
1) estimated core length (cm) 
2) estimate core right/left crop pixel 
3) estimated core top crop pixel 
4) estimated core bottom crop pixel 
can be returned with varargout as a suggested starting location for manual cropping  

REQUIRED ENVIRONMENT 
================================ 
MATLAB 2022b or newer (due to use of estgeotform2d) 
	replace with estimateGeometricTransform for older versions 
	Note that translation parameters are at different matrix indicies  
Computer Vision Toolbox 
Image Processing Toolbox 
Statistics and Machine Learning Toolbox  

EXAMPLES 
================================  

Disable color adjustment and tilt correction 
corealign('folder','color',0,'tilt',0);  

Find new SURF points after tilt correction 
corealign('folder','alt',1);  

Disable tilt correction 
corealign('folder','tilt',0);  

autocrop and return cropped and uncropped images with crop pixel locations 
[P,Pcrop,~,~,CoreDims] = corealign(folder,varargin)  

Note on GPU usage: 
'detectSURFFeatures.m' does not work on GpuArrays, so Gpu support is not implemented 
The following example has feature extraction working on a GPU, but it is actually 
slower than a CPU using 'detectSURFFeatures.m' 
https://www.mathworks.com/help/gpucoder/ug/feature-extraction-using-surf.html  

Adapted from an algorithm developed by Jan Moren. 
S.P Obrochta 7/2023 
