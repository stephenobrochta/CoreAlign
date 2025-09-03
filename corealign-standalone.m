function [P,Pcrop,imgs,Charts,Scale_extracted,varargout] = corealign(folder,varargin)
% 
% [P,Pcrop,imgs,Charts,Scale_extracted,varargout] = corealign(folder,varargin)
% 
% This function aligns a sequence of overlapping images of sediment cores
% using extracted Speeded Up Robust Feature (SURF) points. Features are matched
% between two adjacent images. The translation vector between the matched points
% are used to calculate the location in a panorama for each image. If an object
% such as sample label or color chart is ridigly attached to the camera such that
% it *should* be visible in the same location in each image, then this object
% can be used for tilt correction caused by e.g., camera vibration, which can be
% caused by sliding the camera on rails or ship movement. If this object includes
% a color chart any variable intensity from light flickering can also be corrected.
% 
% REQUIRED INPUT VARIABLES
% ================================
% 'folder' | string containing folder name with image sequence (uint8 or uint16).
%
% Optional INPUT VARIABLES
% ================================

% CoreDims | table containing cropping pixel locations and estimated core length

% OUTPUT VARIABLES
% ================================
% 'P' | Panorama of aligned images as either uint8 or uint16 depending on input images.
% 'imgs' | cell array of all images after color adjustment.
% 'Charts' | cell arry of all color charts extracted from images after color adjustment.
%
% OPTIONAL ARGUMENT / VALUE PAIRS
% ================================
% 
% 'alt' | [1,0] Default: 0
% Alternate translation method is performed if 1. If tilt correction is performed,
% a new set of SURF points are detected and features matched for each pair of images.
% Translation distances are calculated from the new matched pairs of SURF points
% Default method: SURF points are inversely transformed using tilt correction matrix
% Alternate method: After tilt correction of images, a new set of SURF points are obtained
% 
% 'color' | [1,0] Default: 1
% Adjusts color values of all images to match a reference image in the sequence by detecting
% and extracting the color chart from every image. The mean RGB values are calculated for each
% Chart, and the chart with values closest to the median of all becomes the reference.
% This removes variable intensity between images that could be caused by lighting flicker.
% Images of the color charts are stored in /private. Additional images of charts can be added.
% Samples of color charts are contined in /private. Additional images of charts can bee added.
% 
% 'tilt' | [1,0] Default: 1
% Corrects for variable tilt between two image pairs that could be caused by camera movement
% by using the translation vectors of SURF points corresponding to a static object that
% *should* be in the same location in both image pairs. Examples of static objects are
% sample labels and color charts that move rigidly with the camera.
% Manually moved objects should not be used!
% 
% 'fig' | [1,0] Default: 0
% display a figure with progress of alignment. Looks cool I guess but adds time
% 
% 'crop' | [1,0] Default: 0
% Experimental automatic cropping of the image and determination of core length.
% The first (left-most) and last (right-most) images are compared to image templates.
% The first image is compared to a the first 5-cm of the scale The last image is compared
% an image of 3D-printed block which must be placed against the bottom of the core.
% If these features are detected correctly, pixel dimensions of the core are obtained
% allowing it to be segmented from the background. The top and bottom pixel locations of
% the scale bar are also obtained. Next, the scale bar is segmented from the core image
% and compared to a template scale image of a known length and pixel number.
% A geometric transformation is obtained to warp the extracted scalebar to the template.
% Black fill value are used, and the number of non-black pixels in the warped image is used
% to calculate length in centimeters based on the number of pixels / cm in the template image.
% REQUIREMENTS FOR AUTOCROP AND LENGTH DETERMINATION
% The scale and bottom block used when photographing cores must be visually identical to the templates
% The scale template must be as long or longer than the core.
% NOTE: the input parameter 'MaximumDistance' needs to be specified or the geometric transformation 
% will likely fail because individual tick marks all contain similar features. Small changes to the
% maximum distance yield slightly different results.
% VARARGOUT returns an 1x4 table with:
% 1) estimated core length (cm)
% 2) estimate core right/left crop pixel
% 3) estimated core top crop pixel
% 4) estimated core bottom crop pixel
% can be returned with varargout as a suggested starting location for manual cropping
% 
% REQUIRED ENVIRONMENT
% ================================
% MATLAB 2022b or newer (due to use of estgeotform2d)
% 	replace with estimateGeometricTransform for older versions
% 	Note that translation parameters are at different matrix indicies 
% Computer Vision Toolbox
% Image Processing Toolbox
% Statistics and Machine Learning Toolbox
% 
% EXAMPLES
% ================================
% 
% Disable color adjustment and tilt correction
% corealign('folder','color',0,'tilt',0);
% 
% Find new SURF points after tilt correction
% corealign('folder','alt',1);
% 
% Disable tilt correction
% corealign('folder','tilt',0);
% 
% autocrop and return cropped and uncropped images with crop pixel locations
% [P,Pcrop,~,~,CoreDims] = corealign(folder,varargin)
% 
% Note on GPU usage:
% 'detectSURFFeatures.m' does not work on GpuArrays, so Gpu support is not implemented
% The following example has feature extraction working on a GPU, but it is actually
% slower than a CPU using 'detectSURFFeatures.m'
% https://www.mathworks.com/help/gpucoder/ug/feature-extraction-using-surf.html
% 
% Adapted from an algorithm developed by Jan Moren.
% S.P Obrochta 7/2023

tic

% input parser
p = inputParser;
p.KeepUnmatched = true;
p.CaseSensitive = false;
Alt = 0;
Color = 1;
Tilt = 1;
Crop = 0;
Spos = 'top';
ChartNum = 0;

addParameter(p,'alt',Alt);
addParameter(p,'color',Color);
addParameter(p,'tilt',Tilt);
addParameter(p,'crop',Crop);
addParameter(p,'spos',Spos);
addParameter(p,'chartnum',ChartNum);

parse(p,varargin{:});
Alt = p.Results.alt;
Color = p.Results.color;
Tilt = p.Results.tilt;
Crop = p.Results.crop;
Spos = p.Results.spos;
ChartNum = p.Results.chartnum;

% numbers are passed as strings
if ischar(Alt)
	Alt = str2double(Alt);
end
if ischar(Color)
	Color = str2double(Color);
end
if ischar(Tilt)
	Tilt = str2double(Tilt);
end
if ischar(Crop)
	Crop = str2double(Crop);
end
if ischar(ChartNum)
	ChartNum = str2double(ChartNum);
end

% alternate method not needed if no tilt correction
if Tilt == 0
	Alt = 0;
end

% Method to use
if Alt
	str = 'Alternate (extract new SURF points)';
else
	str = 'Default (SURF point transformation)';
end
disp(['Translation Method: ' str])

% read in image data store
imds = imageDatastore(folder);
[~,fnams] = fileparts(imds.Files);
numImages = numel(imds.Files);
imageSizes = zeros(numImages,2);
orientation = nan(size(imds.Files));

% Match point variables
% The ideal number of match points to obtain. Now this is only used for moving points
n = 800;

% distance in pixels plus minus the mode after which match points are considered outliers
d = 25;

% number of pixels plus/minus the mode needed to obtain the desired number of match points
[stepping_moving,stepping_static] = deal(nan(size(imds.Files,1)-1,size(imds.Files,2)));

% cell arrays to hold the SURF point objects that correspond to moving and static match points
[matchedPoints1_moving, ...
matchedPoints2_moving, ... 
matchedPoints1_static, ...
matchedPoints2_static, ...
matchedPoints1_moving_culled, ...
matchedPoints2_moving_culled, ...
matchedPoints1_static_culled, ...
matchedPoints2_static_culled, ...
distance_x_m, ...
distance_y_m, ...
distance_x_s, ...
distance_y_s, ...
] = deal(cell(size(imds.Files,1)-1,size(imds.Files,2)));

% transform points
if ~Alt
	[UV1, UV2] = deal(cell(size(imds.Files,1)-1,size(imds.Files,2)));
end

% Summary stats of distances. 50th, ~2.5, ~97.5 percentiles
[distance_x_med_m, ...
distance_y_med_m, ...
distance_x_med_s, ...
distance_y_med_s ...
] = deal(nan(size(matchedPoints1_moving,1),3));

% Positioning of images in panorama
% relative translation distance and left pixel index for the 2nd image in each pair
[xdist,ydist,LeftEdge] = deal(nan(size(matchedPoints1_moving)));

% position vectors for images calculated from relative translation distance
xpos = ones(numel(imds.Files),2);
ypos = zeros(numel(imds.Files),2);

% median values from each color chart
Charts = cell(numImages,1);
RGB = nan(numImages,1);

% geometric transformation matrices for tilt correction
TformsS(1:numImages) = deal(projtform2d);

% turn off geometric transformation warnings
id = 'MATLAB:nearlySingularMatrix';

% Keep track of the number of things that fail
Failed = 0;

% load images
disp(folder)
disp('Loading images')
imgs = readall(imds);


% 1 photograph rotation


% Confirm exif orientation is correctly set
% This is assuming that the photos were taken in portrait orientation
for i = 1:length(orientation)
	exif = imfinfo(imds.Files{i});
	try
		orientation(i) = exif.Orientation;
		switch orientation(i)
		case 8
			imgs{i} = imrotate(imgs{i},-270);
		case 6 % rotating in the finder sets exif orientation to 6 but image still reads in as unroated
			imgs{i} = imrotate(imgs{i},270);
		case 1 % should be an unrotated portrait image but check
			if size(imgs{i},2) > size(imgs{i},1)
				imgs{i} = imrotate(imgs{i},270);
			end
		end
	catch
		disp([imds.Files{i} ' has no EXIF Orientation property'])
	end
end

disp('Rotating all images based on angle between first and second image')
% get angles for just the first pair
[rotatepts1,rotatepts2,junk1,junk2] = getStaticMovingSURF(imgs{1},imgs{2},n,d);

% Remove angles outside of 95.4 percentile range
rotate_angle = removeOutlyingAngles(rotatepts1,rotatepts2,junk1,junk2);

% Rotate all images the same amount based on the difference between the first and second image
for i = 1:numImages
	if i == 1, rotate_angle = rotate_angle(1) - 180; end
	imgs{i} = imrotate(imgs{i},rotate_angle,'crop');
	imageSizes(i,:) = size(imgs{1},1:2);
end


% 2 get color adjustment values


% First get the color adjustment data and save for the last step because it alters detected features.
if Color	
	disp('Finding and extracting color chart')
	% Points and features in first image
	Ipoints = detectSURFFeatures(rgb2gray(imgs{1}));
	[IFeatures,IValidPoints] = extractFeatures(rgb2gray(imgs{1}),Ipoints);

	% load all color chart templates
	ChartTemplates = imageDatastore(fullfile(fileparts(which('corealign')),'private/charts'));
	ChartTemplates = readall(ChartTemplates);
	
	% Find the chart that matches best with the the first image
	[ChartPoints,ChartFeatures,ChartValidPoints,indexPairs,ChartMatchedPoints] = deal(cell(size(ChartTemplates)));
	nMatched = zeros(size(ChartTemplates));
	if ChartNum == 0
		for i = 1:numel(ChartTemplates)
			ChartPoints{i} = detectSURFFeatures(rgb2gray(ChartTemplates{i}));
			[ChartFeatures{i},ChartValidPoints{i}] = extractFeatures(rgb2gray(ChartTemplates{i}),ChartPoints{i});
			indexPairs{i} = matchFeatures(IFeatures,ChartFeatures{i});
			ChartMatchedPoints{i} = ChartValidPoints{i}(indexPairs{i}(:,2));
			nMatched(i) = ChartMatchedPoints{i}.Count;
		end
	else
		i = ChartNum;
		ChartPoints{i} = detectSURFFeatures(rgb2gray(ChartTemplates{i}));
		[ChartFeatures{i},ChartValidPoints{i}] = extractFeatures(rgb2gray(ChartTemplates{i}),ChartPoints{i});
		indexPairs{i} = matchFeatures(IFeatures,ChartFeatures{i});
		ChartMatchedPoints{i} = ChartValidPoints{i}(indexPairs{i}(:,2));
		nMatched(i) = ChartMatchedPoints{i}.Count;
	end
	% Match the best one to the image
	index = nMatched == max(nMatched);
	indexPairs = indexPairs{index};
	ChartMatchedPoints = ChartMatchedPoints{index};
	ChartTemplates = ChartTemplates{index};
	IMatched = IValidPoints(indexPairs(:,1));	

	% extract charts and calculate mean values
	for i = 1:numImages
		rng('default')
		warning('off',id)
		Charts{i} = imwarp(imgs{i},estgeotform2d(IMatched,ChartMatchedPoints,'similarity'),'OutputView',imref2d(size(ChartTemplates)),'fillvalues',0);
		warning('on',id)

		% Check if something was extracted
		if max(Charts{i}(:)) == 0
			warning(['Failed to extract a color chart from image' fnams{i} '. Rotating color chart'])
			ChartTemplates = imrotate(ChartTemplates,180);
			ChartPoints = detectSURFFeatures(rgb2gray(ChartTemplates));
			[ChartFeatures,ChartValidPoints] = extractFeatures(rgb2gray(ChartTemplates),ChartPoints);
			IndexPairsNew = matchFeatures(IFeatures,ChartFeatures);
			IMatched = IValidPoints(IndexPairsNew(:,1));
			ChartMatchedPoints = ChartValidPoints(IndexPairsNew(:,2));
			rng('default')
			warning('off',id)
			Charts{i} = imwarp(imgs{i},estgeotform2d(IMatched,ChartMatchedPoints,'similarity'),'OutputView',imref2d(size(ChartTemplates)),'fillvalues',0);
			warning('on',id)
			if max(Charts{i}(:)) == 0
				warning("Still failed, Colors cannot be adjusted.")
				Color = 0;
				Failed = Failed + 1;
			break
			end
		end


		% index of nonfill values
		ChartIndex = Charts{i}(:,:,1) > 0 & Charts{i}(:,:,2) > 0 & Charts{i}(:,:,3) > 0;
	
		% Mean values for each color channel
		R = Charts{i}(:,:,1);
		RGB(i,1) = mean(R(ChartIndex));
		G = Charts{i}(:,:,2);
		RGB(i,2) = mean(G(ChartIndex));
		B = Charts{i}(:,:,3);
		RGB(i,3) = mean(B(ChartIndex));
	end
end

if Color
	% index of the image with R, B and G channels closest to the median of all charts
	[~,chrt_index] = min(sum(abs(median(RGB) - RGB),2));
	disp(['Adjusting all images by RGB values from image closest to median intensity (' fnams{chrt_index} ')'])
	
	% calculate the correction but do the actual correction after alignment
	RGBcorr = RGB - RGB(chrt_index,1);
end


% 3 Detect SURF points, perform tilt correction, and calculate translation distances



disp('Detecting, extracting and matching features between image pairs')
for i = 1:numImages - 1
	[matchedPoints1_moving{i}, ...
	matchedPoints2_moving{i}, ...
	matchedPoints1_static{i}, ...
	matchedPoints2_static{i}, ...
	stepping_moving(i), ...
	stepping_static(i) ...
	] = getStaticMovingSURF(imgs{i},imgs{i+1},n,d);

	% remove outlyinig points using angles
	[~, ~, ~, ~, ...
	matchedPoints1_moving_culled{i}, matchedPoints2_moving_culled{i}, ...
	matchedPoints1_static_culled{i}, matchedPoints2_static_culled{i}, ...
	] = removeOutlyingAngles(matchedPoints1_moving{i},matchedPoints2_moving{i}, ...
	matchedPoints1_static{i},matchedPoints2_static{i});
	
	% remove outlying points using x and y distance
	[distance_x_m{i}, distance_y_m{i}, distance_x_s{i}, distance_y_s{i}, ...
	matchedPoints1_moving_culled{i}, matchedPoints2_moving_culled{i}, ...
	matchedPoints1_static_culled{i}, matchedPoints2_static_culled{i}, ...
	distance_x_med_m(i,:), distance_y_med_m(i,:), distance_x_med_s(i,:), distance_y_med_s(i,:) ...
	] = removeOutlyingMatches(matchedPoints1_moving_culled{i}, matchedPoints2_moving_culled{i}, ...
	matchedPoints1_static_culled{i}, matchedPoints2_static_culled{i});

	disp(['Image pair ' num2str(i) '/' num2str(numImages - 1) ': ' fnams{i} ' / ' fnams{i + 1}])

	% tilt correction
	if Tilt
		if matchedPoints1_static_culled{i}.Count < 4
			warning('Too few SURF points for geometric transformation')
			continue
		else
			disp(['     Static SURF point number: ' num2str(matchedPoints1_static_culled{i}.Count) ])
			disp(['     Median movement distance (pixels): ' num2str(distance_x_med_s(i,1)) ' (x); ' ...
			num2str(distance_y_med_s(i,1)) ' (y); '])
			rng('default')
			warning('off',id)
			TformsS(i + 1) = estgeotform2d(matchedPoints2_static_culled{i},matchedPoints1_static_culled{i},'projective');
			outputView = imref2d(size(imgs{i + 1}));
			imgs{i + 1} = imwarp(imgs{i + 1}, TformsS(i + 1),'outputview',outputView);
			warning('on',id)
		end
	end

	if ~Alt
		UV1{i} = [matchedPoints1_moving_culled{i}.Location(:,1), matchedPoints1_moving_culled{i}.Location(:,2)];
		XY2 = [matchedPoints2_moving_culled{i}.Location(:,1), matchedPoints2_moving_culled{i}.Location(:,2)];
		warning('off',id)
		UV2{i} = transformPointsForward(TformsS(i + 1),XY2);
		warning('on',id)

		% Check points
		if isempty(UV1{i}) || isempty(UV2{i})
			warning(['Unable to find valid match between images ' fnams{i} ' and ' fnams{i + 1} '. Switching to alternate alignment method '])
			Alt = 1;
			Failed = Failed + 1;
		end
	end

	if Alt
		% Get a new set of match points after tilt correction image
		rng('default')
		[matchedPoints1_moving{i}, ...
		matchedPoints2_moving{i}, ...
		matchedPoints1_static{i}, ...
		matchedPoints2_static{i}, ...
		stepping_moving(i), ...
		stepping_static(i) ...
		] = getStaticMovingSURF(imgs{i},imgs{i+1},n,d);
	
		% remove outlying points using x and y distance
		[distance_x_m{i}, distance_y_m{i}, distance_x_s{i}, distance_y_s{i}, ...
		matchedPoints1_moving_culled{i}, matchedPoints2_moving_culled{i}, ...
		matchedPoints1_static_culled{i}, matchedPoints2_static_culled{i}, ...
		distance_x_med_m(i,:), distance_y_med_m(i,:), distance_x_med_s(i,:), distance_y_med_s(i,:) ...
		] = removeOutlyingMatches(matchedPoints1_moving{i}, matchedPoints2_moving{i}, ...
		matchedPoints1_static{i}, matchedPoints2_static{i});
		UV1{i} = [matchedPoints1_moving_culled{i}.Location(:,1), matchedPoints1_moving_culled{i}.Location(:,2)];
		UV2{i} = [matchedPoints2_moving_culled{i}.Location(:,1), matchedPoints2_moving_culled{i}.Location(:,2)];

	end

	% translation distances for each image
	xdist(i) = round(prctile(UV1{i}(:,1) - UV2{i}(:,1),50));
	ydist(i) = round(prctile(UV1{i}(:,2) - UV2{i}(:,2),50));

	disp(['     Moving SURF point number: ' num2str(matchedPoints1_moving_culled{i}.Count) ])
	disp(['     ' fnams{i + 1} ' Distance from ' fnams{i} ': ' num2str(xdist(i)) ' (x), ' ...
	num2str(ydist(i)) ' (y) pixels'])
	

end


% 4. Apply the color correction after detecing all SURF points for reproducibility


disp('Creating composite image')
if Color
	for i = 1:numImages
		imgs{i}(:,:,1) = double(imgs{i}(:,:,1)) - RGBcorr(i,1);
		imgs{i}(:,:,2) = double(imgs{i}(:,:,2)) - RGBcorr(i,2);
		imgs{i}(:,:,3) = double(imgs{i}(:,:,3)) - RGBcorr(i,3);
		Charts{i}(:,:,1) = double(Charts{i}(:,:,1)) - RGBcorr(i,1);
		Charts{i}(:,:,2) = double(Charts{i}(:,:,2)) - RGBcorr(i,2);
		Charts{i}(:,:,3) = double(Charts{i}(:,:,3)) - RGBcorr(i,3);
	end
end


% 5. Calculate image overlaps and positions then build panorama


for i = 2:length(ypos)
	ypos(i,1) = ypos(i - 1,1) + ydist(i - 1);
end
% set the bottom most most image position to zero
ypos(:,1) = ypos(:,1) - min(ypos(:,1));

% Add second dimension and move up first dim by one
for i = 1:numel(imgs)
	ypos(i,2) = ypos(i,1) + size(imgs{i},1);
end
ypos(:,1) = ypos(:,1) + 1;

% height of panorama
ysize = max(ypos(:,2));

% initialzied matrix of ones, so position one = 1
xpos(1,2) = size(imgs{1},2);
% set 2:end start positions
xpos(2:end,1) = transpose(cumsum(xdist));
for i = 2:numel(imgs)
	xpos(i,2) = size(imgs{i},2) + xpos(i,1);
	% Mid point between centers of previous and current image
	LeftEdge(i) = round((size(imgs{i},2) / 2) - xdist(i - 1) / 2) + 1;
	xpos(i,1) = xpos(i,1) + LeftEdge(i);
end

% length of panorama
xsize = xpos(end,2);

% initialize white panorma
if isa(imgs{1},'uint8')
	P = uint8(255 * ones(ysize,xsize,3));
else
	P = uint16(65535 * ones(ysize,xsize,3));
end

% first image
P(ypos(1,1):ypos(1,2),xpos(1,1):xpos(1,2),:) = imgs{1};

% remaining images
for i = 2:numel(imgs)
	P(ypos(i,1):ypos(i,2),xpos(i,1):xpos(i,2),:) = imgs{i}(:,LeftEdge(i):end,:);
end


% 6. Determine locations to crop core top and bottom


% Cropping matches features between the first (core top) and last (core bottom) images
% and templates stored in this functions "private" folder. Objects in the core photos 
% must be identical to the templates.
% Crop = 1 | crop top and bottom and outer scalebar edge
% Crop = 2 | crop only bottom
% Crop = 3 | crop only top and outer scalebar edge
if Crop
	% First try and find the scale in the first image to index core top
	disp('Finding core zero point in first image')
	S = imread('private/L5PxCM200.png');
	ptsS = detectSURFFeatures(rgb2gray(S));
	ptsI = detectSURFFeatures(rgb2gray(imgs{1}));
	[featuresS,validPtsS] = extractFeatures(rgb2gray(S),ptsS);
	[featuresI,validPtsI] = extractFeatures(rgb2gray(imgs{1}),ptsI);
	indexPairsI = matchFeatures(featuresI,featuresS);
	matchedI = validPtsI(indexPairsI(:,1));
	matchedS = validPtsS(indexPairsI(:,2));
	rng('default')
	warning('off',id)
	[TformScale,~,~] = estgeotform2d(matchedI,matchedS,'similarity');
	outputView = imref2d(size(S));
	S_extracted = imwarp(imgs{1},TformScale,'OutputView',outputView,'fillvalues',[0,0,0]);
	warning('on',id)
	% check if extraction worked and reduce distance
	if ssim(S_extracted,S) < 0.55 
		disp('Zero point not detected Reducing maximum distance.')
		D = matchedI.Location - matchedS.Location(:,2);
		D = hypot(D(:,1),D(:,2));
		MaxD = max(D);
		while ssim(S_extracted,S) < 0.55
			warning('off',id)
			[TformScale,~,~] = estgeotform2d(matchedI,matchedS,'similarity','MaxD',MaxD);
			S_extracted = imwarp(imgs{1},TformScale,'OutputView',outputView,'fillvalues',[0,0,0]);
			warning('on',id)
			MaxD = MaxD - 100;
			
			% give up at some point
			if MaxD < 0
				break
			end
		end
	end
	
	if ssim(S_extracted,S) < 0.55
		Crop = 2;
		warning('Failed to detect zero point. Cannot crop composite image to core top')
		[ScaleRight,ScaleEdge] = deal(1);
		Failed = Failed + 1;
	end

	disp('Detecting core bottom block in last image')
	T = imread('private/CoreBottomTarget.JPG');
	ptsT = detectSURFFeatures(rgb2gray(T));
	ptsI = detectSURFFeatures(rgb2gray(imgs{end}));
	[featuresT,validPtsT] = extractFeatures(rgb2gray(T),ptsT);
	[featuresI,validPtsI] = extractFeatures(rgb2gray(imgs{end}),ptsI);
	indexPairsI = matchFeatures(featuresI,featuresT);
	matchedI = validPtsI(indexPairsI(:,1));
	matchedT = validPtsT(indexPairsI(:,2));
	
	rng('default')
	% 3rd output is status and when output 'estgeotform2d' won't error out
	warning('off',id)
	[TformBlock,~,~] = estgeotform2d(matchedI,matchedT,'similarity');
	outputView = imref2d(size(T));
	T_extracted = imwarp(imgs{end},TformBlock,'OutputView',outputView,'fillvalues',[0,0,0]);
	warning('on',id)
	% check if extraction worked and reduce distance
	if ssim(T_extracted,T) < 0.55 
		disp('Block not detected Reducing maximum distance.')
		D = matchedI.Location - matchedT.Location(:,2);
		D = hypot(D(:,1),D(:,2));
		MaxD = max(D);
		while ssim(T_extracted,T) < 0.55 
			warning('off',id)
			[TformBlock,~,~] = estgeotform2d(matchedI,matchedT,'similarity','MaxD',MaxD);
			T_extracted = imwarp(imgs{end},TformBlock,'OutputView',outputView,'fillvalues',[0,0,0]);
			warning('on',id)
			MaxD = MaxD - 100;
			
			% give up at some point
			if MaxD < 0
				break
			end
		end
	end

	% rotate and try again
	if ssim(T_extracted,T) < 0.55 
		disp('Block not detected. Rotating')
		T = imrotate(T,180);
		ptsT = detectSURFFeatures(rgb2gray(T));
		ptsI = detectSURFFeatures(rgb2gray(imgs{end}));
		[featuresT,validPtsT] = extractFeatures(rgb2gray(T),ptsT);
		[featuresI,validPtsI] = extractFeatures(rgb2gray(imgs{end}),ptsI);
		indexPairsI = matchFeatures(featuresI,featuresT);
		matchedI = validPtsI(indexPairsI(:,1));
		matchedT = validPtsT(indexPairsI(:,2));
		rng('default')
		warning('off',id)
		[TformBlock,~,~] = estgeotform2d(matchedI,matchedT,'similarity');
		outputView = imref2d(size(T));
		T_extracted = imwarp(imgs{end},TformBlock,'OutputView',outputView,'fillvalues',[0,0,0]);
		warning('on',id)

		% check if extraction worked and reduce distance
		if ssim(T_extracted,T) < 0.55 
			disp('Block not detected Reducing maximum distance.')
			D = matchedI.Location - matchedT.Location(:,2);
			D = hypot(D(:,1),D(:,2));
			MaxD = max(D);
			while ssim(T_extracted,T) < 0.55
				warning('off',id)
				[TformBlock,~,~] = estgeotform2d(matchedI,matchedT,'similarity','MaxD',MaxD);
				T_extracted = imwarp(imgs{end},TformBlock,'OutputView',outputView,'fillvalues',[0,0,0]);
				warning('on',id)
				MaxD = MaxD - 100;
				
				% give up at some point
				if MaxD < 0
					break
				end
			end
		end
	end
	if ssim(T_extracted,T) < 0.55
		if Crop == 2
			Crop = 0;
			warning('Also failed to find block. Not cropping image')
		else
			Crop = 3;
			warning('Failed to find core bottom. Cannot crop composite image to core bottom')
			CoreBottom = size(imgs{end},2);
		end
		Failed = Failed + 1;
	end
end

% If core top was detected set indices in first image
if Crop == 1 || Crop == 3
	% transform matrix is relative to last image after slight rotation etc to match template
	warning('off',id)
	TformScaleinv = invert(TformScale);
	warning('on',id)
	ScaleRight = round(abs(TformScaleinv.A(1,3)));
	ScaleEdge = round(abs(TformScaleinv.A(2,3)));

	% final check for invalid scale detection
	if ScaleEdge < 1
		disp(['Core top crop pixel location = ' num2str(ScaleEdge)])
		Crop = 2;
	end
end

% If bottom was detected
if Crop == 1 || Crop == 2
	warning('off',id)
	TformBlockinv = invert(TformBlock);
	warning('on',id)
	% Check block rotation
	if abs(TformBlock.RotationAngle) > 90
		% Block was rotated to match template so position is right side. Subtract template width
		CoreBottom = round(abs(TformBlockinv.A(1,3))) - size(T,2);
	else
		% Block not rotated so left side detected. Use as is
		CoreBottom = round(abs(TformBlockinv.A(1,3)));
	end
end


% 7. Segment the core from background


if Crop
	% Adjust core bottom crop information based on panoraama size
	CoreBottom = size(P,2) - (size(imgs{end},2) - CoreBottom);
	
	% crop top and bottom of core
	if strcmp(Spos,'top') && (Crop == 1 || Crop == 3)
		% crop from the scale edge downward
		Pcrop = P(ScaleEdge:end,ScaleRight:CoreBottom,:);
	elseif strcmp(Spos,'bot') && (Crop == 1 || Crop == 3)
		% crop from the scale edge upward
		Pcrop = P(1:ScaleEdge,ScaleRight:CoreBottom,:);
	else
		% Just the bottom (ScaleRight = 1)
		Pcrop = P(:,ScaleRight:CoreBottom,:);
	end

	% Isolate scale to make extraction more reliable
	Scale_extracted = Pcrop(1:size(S,1),:,:);
end

if Crop == 1
	% extract scale
	disp('Extracting scalebar to estimate core length')
	Scale = imread('private/L109_3PxCM200.png');
	ptsI = detectSURFFeatures(rgb2gray(Scale_extracted));
	ptsScale = detectSURFFeatures(rgb2gray(Scale));
	[featuresI,validPtsI] = extractFeatures(rgb2gray(Scale_extracted),ptsI);
	[featuresScale,validPtsScale] = extractFeatures(rgb2gray(Scale),ptsScale);
	indexPairsI1 = matchFeatures(featuresI,featuresScale);
	matchedI1 = validPtsI(indexPairsI1(:,1));
	matchedScale = validPtsScale(indexPairsI1(:,2));
	
	% match scale and extract starting with reasonable max distance for 1 meter section
	rng('default')
	MaxD = 1500;
	warning('off',id)
	[TformFullScale,inlierIdx,~] = estgeotform2d(matchedI1,matchedScale,'similarity','MaxDistance',MaxD);
	outputView = imref2d(size(Scale));
	Scale_extracted = imwarp(Pcrop,TformFullScale,'OutputView',outputView,'fillvalues',[0,0,0]);
	warning('on',id)
	if ssim(Scale_extracted,Scale) < 0.3
		disp('Scalebar not extracted. Retrying.')
		D = matchedI1.Location(inlierIdx,:) - matchedScale.Location(inlierIdx,:);
		D = hypot(D(:,1),D(:,2));
		while ssim(Scale_extracted,Scale) < 0.3 || sum(inlierIdx) == 0
			warning('off',id)
			[TformFullScale,inlierIdx,~] = estgeotform2d(matchedI1,matchedScale,'similarity','MaxDistance',MaxD);
			Scale_extracted = imwarp(Pcrop,TformFullScale,'OutputView',outputView,'fillvalues',[0,0,0]);
			warning('on',id)
			MaxD = MaxD + 100;
			if MaxD > max(D)
				break
			end
		end
	end
	if ssim(Scale_extracted,Scale) < 0.3 || sum(inlierIdx) == 0
		disp('Failed to extract scalebar. Not estimating core length')
		CoreLength = NaN;
		varargout = {[CoreLength,ScaleEdge,ScaleRight,CoreBottom]};
	else
		% Estimate Length
		Black = Scale_extracted(:,:,1) == 0 & Scale_extracted(:,:,2) == 0 & Scale_extracted(:,:,3) == 0;
		index = nan(size(Black,1),1);
		for i = 1:size(Scale,1)
			index(i) = find(Black(i,:) == 0,1,'last');
		end
		CoreLength = max(index) / 200;
		disp(['Estimated core length: ' num2str(CoreLength) ' cm'])
		varargout = {array2table([CoreLength,ScaleEdge,ScaleRight,CoreBottom],'VariableNames',{'Length','Top','Edge','Bottom'})};	
	end
end

% Variables related to cropping might not be assigned
if ~exist('Pcrop','var')
	Pcrop = [];
end
if ~exist('varargout','var')
	varargout = {[]};
end
if ~exist('Scale_extracted','var')
	Scale_extracted = [];
end

if Failed > 0
	cmdstr = strrep(strjoin(["corealign(" "'" folder "'" ",'alt'," num2str(Alt) ",'color',0,'crop'," num2str(Crop) ",'fig'," num2str(Fig) ");"]),' ','');
	disp([num2str(Failed) ' Detection(s) failed. Core not aligned according to input parameters'])
	disp('For efficiency, use the below command if re-running')
	disp(cmdstr)
end

toc
[~,fnam] = fileparts(folder);
imwrite(P,[fnam '.jpg'])
if ~isempty(Pcrop)
	imwrite(Pcrop,[fnam '_crop.jpg'])
end
