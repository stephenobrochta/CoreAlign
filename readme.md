[P,imgs,Charts] = corealign(folder,varargin)

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

'folder' | string containing folder name with image sequence (uint8 or uint16).

OUTPUT VARIABLES

'P' | Panorama of aligned images as either uint8 or uint16 depending on input images.
'imgs' | cell array of all images after color adjustment.
'Charts' | cell arry of all color charts extracted from images after color adjustment.

OPTIONAL ARGUMENT / VALUE PAIRS

'alt' | [1,0] Default: 0
Alternate translation method is performed if 1. If tilt correction is performed,
a new set of SURF points are detected and features matched for each pair of images.
Translation distances are calculated from the new matched pairs of SURF points

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

Default method: SURF points are inversely transformed using tilt correction matrix
Alternate method: After tilt correction of images, a new set of SURF points are obtained

REQUIRED ENVIRONMENT

MATLAB 2022b or newer (due to use of estgeotform2d)
	replace with estimateGeometricTransform for older versions
Computer Vision Toolbox
Image Processing Toolbox
Statistics and Machine Learning Toolbox

EXAMPLES

Disable color adjustment and tilt correction
corealign('folder','color',0,'tilt',0);

Find new SURF points after tilt correction
corealign('folder','alt',1);

Disable tilt correction
corealign('folder','tilt',0);

Note on GPU usage:
'detectSURFFeatures.m' does not work on GpuArrays, so Gpu support is not implemented
The following example has feature extraction working on a GPU, but it is actually
slower than a CPU using 'detectSURFFeatures.m'
https://www.mathworks.com/help/gpucoder/ug/feature-extraction-using-surf.html

Adapted from an algorithm developed by Jan Moren.
S.P Obrochta 4/2023