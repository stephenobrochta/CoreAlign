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
% VARARGOUT returns an 1x4 array with:
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
Fig = 0;
Crop = 0;

addParameter(p,'alt',Alt,@isnumeric);
addParameter(p,'color',Color,@isnumeric);
addParameter(p,'tilt',Tilt,@isnumeric);
addParameter(p,'fig',Fig,@isnumeric);
addParameter(p,'crop',Crop,@isnumeric);

parse(p,varargin{:});
Alt = p.Results.alt;
Color = p.Results.color;
Tilt = p.Results.tilt;
Fig = p.Results.fig;
Crop = p.Results.crop;

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

% load images
disp(folder)
disp('Loading images')
imgs = readall(imds);


% 0 photograph rotation


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


% 1. Autocrop - do this first so it errors out quickly


% Cropping matches features between the first (core top) and last (core bottom) images
% and templates stored in this functions "private" folder. Objects in the core photos 
% must be identical to the templates.
% Crop = 1 | crop top and bottom
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
	if ssim(S_extracted,S) < 0.5 
		disp('Zero point not detected Reducing maximum distance.')
		D = matchedI.Location - matchedS.Location(:,2);
		D = hypot(D(:,1),D(:,2));
		MaxD = max(D);
		while ssim(S_extracted,S) < 0.5
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
	if ssim(S_extracted,S) < 0.5
		Crop = 2;
		disp('Failed to detect zero point. Cannot crop composite image to core top')
		[ScaleRight,ScaleTop] = deal(1);
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
	if ssim(T_extracted,T) < 0.5 
		disp('Block not detected Reducing maximum distance.')
		D = matchedI.Location - matchedT.Location(:,2);
		D = hypot(D(:,1),D(:,2));
		MaxD = max(D);
		while ssim(T_extracted,T) < 0.5 
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
	if ssim(T_extracted,T) < 0.5 
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
		if ssim(T_extracted,T) < 0.5 
			disp('Block not detected Reducing maximum distance.')
			D = matchedI.Location - matchedT.Location(:,2);
			D = hypot(D(:,1),D(:,2));
			MaxD = max(D);
			while ssim(T_extracted,T) < 0.5
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
	if ssim(T_extracted,T) < 0.5
		if Crop == 2
			Crop = 0;
			disp('Also failed to find block. Not cropping image')
		else
			Crop = 3;
			disp('Failed to find core bottom. Cannot crop composite image to core bottom')
			CoreBottom = NaN;
		end
	end
end

% If core top and bottom were detected set indices in first and last image
if Crop == 1
	% transform matrix is relative to last image after slight rotation etc to match template
	warning('off',id)
	tform2inv = invert(TformBlock);
	TformScaleinv = invert(TformScale);
	warning('on',id)
	ScaleRight = round(abs(TformScaleinv.A(1,3)));
	ScaleTop = round(abs(TformScaleinv.A(2,3)));

	% Check block rotation
	if abs(TformBlock.RotationAngle) > 90
		% Block was rotated to match template so position is right side. Subtract template width
		CoreBottom = round(abs(tform2inv.A(1,3))) - size(T,2);
	else
		% Block not rotated so left side detected. Use as is
		CoreBottom = round(abs(tform2inv.A(1,3)));
	end
end


% 2 color adjustment


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
	for i = 1:numel(ChartTemplates)
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
		Charts{i} = imwarp(imgs{i},estgeotform2d(IMatched,ChartMatchedPoints,'similarity'),'OutputView',imref2d(size(ChartTemplates)),'fillvalues',0);

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
			Charts{i} = imwarp(imgs{i},estgeotform2d(IMatched,ChartMatchedPoints,'similarity'),'OutputView',imref2d(size(ChartTemplates)),'fillvalues',0);
			if max(Charts{i}(:)) == 0
				cmdstr = strrep(strjoin(["corealign(" folder ",'alt'," num2str(Alt) ",'color',0,'crop'," num2str(Crop) ",'fig'," num2str(Fig) ");"]),' ','');
				warning(strjoin(["Still failed, skipping. Consider re-running as:" cmdstr]))
				Color = 0;
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

	
	% index of the image with R, B and G channels closest to the median of all charts
	[~,index] = min(sum(abs(median(RGB) - RGB),2));
	disp(['Adjusting all images by RGB values from image closest to median intensity (' fnams{index} ')'])
	
	% calculate the correction but do the actual correction after alignment
	RGBcorr = RGB - RGB(index,1);
end


% 3 rotation based on the angle between the first two images


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


% 4 Detect SURF points


if Fig
	scrnsze = get(0,'MonitorPositions');
	axisargs = {'tickdir','out','color','none','box','on'};
	figure('position',scrnsze(1,:),'color','k')
	drawnow
	scrnsze = get(gcf,'position');
	if Color
		h_img = scrnsze(4) * .6;
		h_chart = scrnsze(4) * .2;
		w_chart = h_chart * (size(Charts{1},2) / size(Charts{1},1));
		b_chart = h_img + scrnsze(4) * .1;
	else
		h_img = scrnsze(4);
	end
	w_img = h_img * (size(imgs{1},2) / size(imgs{1},1));
	if Color
		l_chart = (w_img - w_chart) / 2;
	end

	% graph dimensions based on specified options
	b = scrnsze(4) * 0.05;
	l = (w_img * 2) * 1.075;
	if sum([Tilt Color]) 
		% three columns of figures
		w = (scrnsze(3) - l) * 0.3;
	else
		% one column and three row of figures for moving points only
		w = (scrnsze(3) - l) * 0.9;
		h = scrnsze(4) * 0.25;
	end
	if sum([Tilt Color]) == 2
		% three rows of figures
		h = scrnsze(4) * 0.25;
	elseif sum([Tilt Color]) == 1
		% two rows of figures
		h = scrnsze(4) * 0.4;
	end

end

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
	if Tilt
		if matchedPoints1_static_culled{i}.Count < 4
			warning('Too few SURF points for geometric transformation')
			continue
		else
			disp(['     Static SURF point number: ' num2str(matchedPoints1_static_culled{i}.Count) ])
			disp(['     Median movement distance (pixels): ' num2str(distance_x_med_s(i,1)) ' (x); ' ...
			num2str(distance_y_med_s(i,1)) ' (y); '])
			rng('default')
			TformsS(i + 1) = estgeotform2d(matchedPoints2_static_culled{i},matchedPoints1_static_culled{i},'projective');
			outputView = imref2d(size(imgs{i + 1}));
			imgs{i + 1} = imwarp(imgs{i + 1}, TformsS(i + 1),'outputview',outputView);
		end
	end

	if ~Alt
		UV1{i} = [matchedPoints1_moving_culled{i}.Location(:,1), matchedPoints1_moving_culled{i}.Location(:,2)];
		XY2 = [matchedPoints2_moving_culled{i}.Location(:,1), matchedPoints2_moving_culled{i}.Location(:,2)];
		UV2{i} = transformPointsForward(TformsS(i + 1),XY2);
		
		% Check points
		if isempty(UV1{i}) || isempty(UV2{i})
			warning(['Unable to find valid match between images ' fnams{i} ' and ' fnams{i + 1} '. Switching to alternate alignment method '])
			Alt = 1;
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
	
	if Fig
		% calculate angles
		x1_m = UV1{i}(:,1);
		y1_m = UV1{i}(:,2);
		x2_m = UV2{i}(:,1);
		y2_m = UV2{i}(:,2);
		
		x1_s = matchedPoints1_static_culled{i}.Location(:,1);
		y1_s = matchedPoints1_static_culled{i}.Location(:,2);
		x2_s = matchedPoints2_static_culled{i}.Location(:,1);
		y2_s = matchedPoints2_static_culled{i}.Location(:,2);
		
		angles_m = atan2d(y2_m - y1_m, x2_m - x1_m);
		angles_s = atan2d(y2_s - y1_s, x2_s - x1_s);	

		% image 1
		if exist('H1','var')
			delete(H1)
		end
		H1 = axes('units','pixels','position',[1, b, w_img, h_img]);
		image(imgs{i})
		hold(gca,'on')
		scatter(UV1{i}(:,1),UV1{i}(:,2),'markeredgecolor','r')
		if Tilt
			scatter(matchedPoints1_static_culled{i}.Location(:,1),matchedPoints1_static_culled{i}.Location(:,2),'markeredgecolor','c')
		end
		set(H1,axisargs{:},'xtick',[],'ytick',[],'ydir','reverse','xcolor','w','ycolor','w')
		title(strrep(fnams{i},'_','-'),'color','w')
		drawnow
		
		% chart 1
		if Color
			if exist('H2','var')
				delete(H2)
			end
			H2 = axes('units','pixels','position',[l_chart, b_chart, w_chart, h_chart]);
			imshow(Charts{i})
			title([strrep(fnams{i},'_','-') ' Color Chart'],'color','w')
			drawnow
		end

		% image 2
		if exist('H3','var')
			delete(H3)
		end
		H3 = axes('units','pixels','position',[1 + w_img, b, w_img, h_img]);
		image(imgs{i + 1})
		set(H3,axisargs{:},'xtick',[],'ytick',[],'ydir','reverse','xcolor','w','ycolor','w')
		hold(gca,'on')
		scatter(UV2{i}(:,1),UV2{i}(:,2),'markeredgecolor','r')
		if Tilt
			scatter(matchedPoints1_static_culled{i}.Location(:,1),matchedPoints1_static_culled{i}.Location(:,2),'markeredgecolor','c')
		end
		title(strrep(fnams{i + 1},'_','-'),'color','w')
		drawnow
		
		% chart 2
		if Color
			if exist('H4','var')
				delete(H4)
			end
			H4 = axes('units','pixels','position',[l_chart + w_img, b_chart, w_chart, h_chart]);
			imshow(Charts{i + 1})
			title([strrep(fnams{i + 1},'_','-') ' Color Chart'],'color','w')
			drawnow
		end

		% lines connecting SURF points
		if exist('Himages','var')
			delete(Himages)
		end
		Himages = axes('units','pixels','position',[1, b, 2 * w_img, h_img]);
		hold(Himages,'on')
		for j = 1:length(x1_m)
			line([x1_m(j), x2_m(j) + size(imgs{i},2)],[y1_m(j), y2_m(j)],'color',[1,.75,.75])
		end
		if Tilt
			for j = 1:length(x1_s)
				line([x1_s(j), x2_s(j) + size(imgs{i},2)],[y1_s(j), y2_s(j)],'color',[.75,.75,1])
			end
		end
		set(Himages,axisargs{:},'xtick',[],'ytick',[],'ydir','reverse','xcolor','w','ycolor','w','xlim',[1 2 * size(imgs{i},2)],'ylim',[1 size(imgs{i},1)])
		axis off
		drawnow

		% Moving points X dist
		if exist('H5','var')
			delete(H5)
		end
		H5 = axes('units','pixels','position',[l, b, w, h]);
		hold(H5,'on')
		stairs(2:i+1,xdist(1:i),'-or','linewidth',1)
		set(H5,axisargs{:},'xtick',2:numImages,'xlim',[2 numImages],'xcolor','w','ycolor','w')
		title('Translation horizontal distance','color','w')
		ylabel('Pixels')
		xlabel('Image Number')
		drawnow
		TickLength = get(H5,'TickLength');

		% Moving points Y dist
		if exist('H6','var')
			delete(H6)
		end
		if sum([Tilt Color]) 
			H6 = axes('units','pixels','position',[l + w, b, w, h]);
		else
			H6 = axes('units','pixels','position',[l, b + h * 1.2, w, h]);
		end
		hold(H6,'on')
		stairs(2:i+1,ydist(1:i),'-or','linewidth',1)
		set(H6,axisargs{:},'xtick',2:numImages,'xlim',[2 numImages],'xcolor','w','ycolor','w')
		if sum([Tilt Color])
			set(gca,'yaxislocation','right')
		end
		title('Translation vertical distance','color','w')
		ylabel('Pixels')
		if sum([Tilt Color])
			xlabel('Image Number')
		end
		drawnow

		% Moving points angle
		if exist('H7','var')
			delete(H7)
		end
		if sum([Tilt Color]) 
			axes('units','pixels','position',[l + 2.25 * w, b, w, h]);
		else
			axes('units','pixels','position',[l, 2 * (b + h) * 1.1, w, h]);
		end
		polarhistogram(deg2rad(abs(angles_m)),'facecolor','r','facealpha',1,'edgecolor','r');
		H7 = gca;
		set(H7,axisargs{:})
		H7.RColor = 'w';
		H7.GridColor = 'w';
		H7.GridAlpha = 1;
		H7.ThetaColor = 'w';
		title('Translation Angle','color','w')
		drawnow
		
		if Tilt		
			% Static points X dist
			if exist('H8','var')
				delete(H8)
			end
			H8 = axes('units','pixels','position',[l, b + h * 1.2, w, h]);
			hold(H8,'on')
			stairs(2:i+1,distance_x_med_s(1:i),'-oc','linewidth',1)
			set(H8,axisargs{:},'xtick',2:numImages,'xlim',[2 numImages],'xcolor','w','ycolor','w')
			title('Tilt-correction horizontal distance','color','w')
			ylabel('Pixels')
			drawnow
	
			% Staic points Y dist
			if exist('H9','var')
				delete(H9)
			end
			H9 = axes('units','pixels','position',[l + w, b + h * 1.2, w, h]);
			hold(H9,'on')
			stairs(2:i+1,distance_y_med_s(1:i),'-oc','linewidth',1)
			set(H9,axisargs{:},'xtick',2:numImages,'xlim',[2 numImages],'yaxislocation','right','xcolor','w','ycolor','w')
			title('Tilt-correction vertical distance','color','w')
			ylabel('Pixels')
			drawnow

			% Static points angle
			if exist('H10','var')
				delete(H10)
			end
			axes('units','pixels','position',[l + 2.25 * w, b + h * 1.2, w, h]);
			polarhistogram(deg2rad(angles_s),'facecolor','c','facealpha',1,'edgecolor','k');
			H10 = gca;
			set(H10,axisargs{:})
			H10.RColor = 'w';
			H10.GridColor = 'w';
			H10.GridAlpha = 1;
			H10.ThetaColor = 'w';
			title('Tilt Angle','color','w')
			drawnow
		end

		if Color
			if exist('H11','var')
				delete(H11)
			end
			if Tilt
				H11 = axes('units','pixels','position',[l, 2 * (b + h) * 1.2, 2 * w, h]);
			else
				H11 = axes('units','pixels','position',[l, (b + h) * 1.2, (scrnsze(3) - l) * 0.9, h]);
			end
			hold(gca,'on')
			stairs(RGB(1:i,1),'-or','linewidth',1)
			stairs(RGB(1:i,2),'-og','linewidth',1)
			stairs(RGB(1:i,3),'-oc','linewidth',1)
			set(H11,axisargs{:},'xtick',1:numImages,'xlim',[1 numImages],'ticklength',TickLength,'xcolor','w','ycolor','w')
			title('Chart color values','color','w')
			ylabel('Color')
			drawnow

			% histogram
			if exist('H12','var')
				delete(H12)
			end
			if Tilt
				H12 = axes('units','pixels','position',[l + 2.2 * w, 2 * (b + h) * 1.2, w, h]);
			else
				H12 = axes('units','pixels','position',[l + 2.2 * w, (b + h) * 1.2, w, h]);
			end
			histogram(Charts{i}(:),'facecolor','c','edgecolor','w')
			set(H12,axisargs{:},'ticklength',TickLength,'xcolor','w','ycolor','w')
			axis tight
			title([strrep(fnams{i},'_','-') ' Color Chart'],'color','w')
			drawnow

			if i == numImages - 1
				delete(H11)
				if Tilt
					H11 = axes('units','pixels','position',[l, 2 * (b + h) * 1.2, 2 * w, h]);
				else
					H11 = axes('units','pixels','position',[l, (b + h) * 1.2, (scrnsze(3) - l) * 0.9, h]);
				end
				hold(gca,'on')
				stairs(RGB(:,1),'-or')
				stairs(RGB(:,2),'-og')
				stairs(RGB(:,3),'-oc')
				set(H11,axisargs{:},'xtick',1:numImages,'xlim',[1 numImages],'ticklength',TickLength,'xcolor','w','ycolor','w')
				title('Chart color values','color','w')
				ylabel('Color')
				drawnow

				% histogram
				if exist('H12','var')
					delete(H12)
				end
				if Tilt
					H12 = axes('units','pixels','position',[l + 2.2 * w, 2 * (b + h) * 1.2, w, h]);
				else
					H12 = axes('units','pixels','position',[l + 2.2 * w, (b + h) * 1.2, w, h]);
				end
				histogram(Charts{i + 1}(:),'facecolor','c','edgecolor','w')
				set(H12,axisargs{:},'ticklength',TickLength,'xcolor','w','ycolor','w')
				axis tight
				title([strrep(fnams{i + 1},'_','-') ' Color Chart'],'color','w')
				drawnow
			end
		end
	end
end


% 2. color adjustment. Apply the color correction after detecing all SURF points for reproducibility


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


% 5. Calculate translation distances and build panorama


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


% 6. Segment the core from background


if Crop
	% Adjust core bottom crop information based on panoraama size
	CoreBottom = size(P,2) - (size(imgs{end},2) - CoreBottom);
	
	% crop top and bottom of core
	Pcrop = P(ScaleTop:end,ScaleRight:CoreBottom,:);

	% Isolate scale to make extraction more reliable
	Scale_extracted = Pcrop(1:size(S,1),:,:);

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
			disp(round(MaxD))
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
		varargout = {[CoreLength,ScaleTop,ScaleRight,CoreBottom]};
	else
		% Estimate Length
		Black = Scale_extracted(:,:,1) == 0 & Scale_extracted(:,:,2) == 0 & Scale_extracted(:,:,3) == 0;
		index = nan(size(Black,1),1);
		for i = 1:size(Scale,1)
			index(i) = find(Black(i,:) == 0,1,'last');
		end
		CoreLength = max(index) / 200;
		disp(['Estimated core length: ' num2str(CoreLength) ' cm'])
		varargout = {[CoreLength,ScaleTop,ScaleRight,CoreBottom]};	
	end
end

if Fig
	figure
	imshow(P)
	if Crop
		figure
		imshow(Pcrop)
	end
end

toc