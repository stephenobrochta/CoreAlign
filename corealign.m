function [P,imgs,Charts] = corealign(folder,varargin)
% 
% [P,imgs,Charts] = corealign(folder,varargin)
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
% Default method: SURF points are inversely transformed using tilt correction matrix
% Alternate method: After tilt correction of images, a new set of SURF points are obtained
% 
% REQUIRED ENVIRONMENT
% ================================
% MATLAB 2022b or newer (due to use of estgeotform2d)
% 	replace with estimateGeometricTransform for older versions
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
% Note on GPU usage:
% 'detectSURFFeatures.m' does not work on GpuArrays, so Gpu support is not implemented
% The following example has feature extraction working on a GPU, but it is actually
% slower than a CPU using 'detectSURFFeatures.m'
% https://www.mathworks.com/help/gpucoder/ug/feature-extraction-using-surf.html
% 
% Adapted from an algorithm developed by Jan Moren.
% S.P Obrochta 4/2023

tic

% input parser
p = inputParser;
p.KeepUnmatched = true;
p.CaseSensitive = false;
Alt = 0;
Color = 1;
Tilt = 1;
Fig = 0;

addParameter(p,'alt',Alt,@isnumeric);
addParameter(p,'color',Color,@isnumeric);
addParameter(p,'tilt',Tilt,@isnumeric);
addParameter(p,'fig',Fig,@isnumeric);

parse(p,varargin{:});
Alt = p.Results.alt;
Color = p.Results.color;
Tilt = p.Results.tilt;
Fig = p.Results.fig;

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
foldern = length(imds.Folders{:});
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
Charts ...
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
% relative translation distance for the 2nd image in each pair
[xdist,ydist] = deal(nan(size(matchedPoints1_moving)));

% position vectors for images calculated from relative translation distance
xpos = ones(numel(imds.Files),2);
ypos = zeros(numel(imds.Files),2);

% median values from each color chart
RGB = nan(numImages,1);

% geometric transformation matrices for tilt correction
% tforms(numImages) = projective2d(eye(3));
tforms(1:numImages) = deal(projtform2d);
% load images
disp(folder)
disp('Loading images')
imgs = readall(imds);

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


% 0 color adjustment


if Color	
	disp('Finding and extracting color chart')
	% Points and features in first image
	Ipoints = detectSURFFeatures(rgb2gray(imgs{1}));
	[IFeatures,IValidPoints] = extractFeatures(rgb2gray(imgs{1}),Ipoints);

	% load all color chart templates
	ChartTemplates = imageDatastore(fullfile(fileparts(which('corealign')),'private'));
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
	% ChartPoints = ChartPoints{index};
	ChartFeatures = ChartFeatures{index};
	ChartValidPoints = ChartValidPoints{index};
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
			warning(['Failed to extract a color chart from image' fnams{i} '. Detecting new SURF points'])
			Ipoints = detectSURFFeatures(rgb2gray(imgs{1}));
			[IFeatures,IValidPoints] = extractFeatures(rgb2gray(imgs{1}),Ipoints);
			IndexParisNew = matchFeatures(IFeatures,ChartFeatures);
			IMatched = IValidPoints(IndexParisNew(:,1));
			ChartMatchedPoints = ChartValidPoints(IndexParisNew(:,2));
			rng('default')
			Charts{i} = imwarp(imgs{i},estgeotform2d(IMatched,ChartMatchedPoints,'similarity'),'OutputView',imref2d(size(ChartTemplates)),'fillvalues',0);
			if max(Charts{i}(:)) == 0
				warning('Still failed, skipping.')
			continue
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

% 1 rotation based on the angle between the first two images


disp('1. Rotation')
% get angles for just the first pair
[rotatepts1,rotatepts2,junk1,junk2] = getStaticMovingSURF(imgs{1},imgs{2},n,d);

% Remove angles outside of 95.4 percentile range
rotate_angle = removeOutlyingAngles(rotatepts1,rotatepts2,junk1,junk2);

% Rotate all images the same amount based on the difference between the first and second image
for i = 1:numImages
	if i == 1, rotate_angle = rotate_angle(1) - 180; end
	imgs{i} = imrotate(imgs{i},rotate_angle,'crop');
	if i == numImages - 1
		imgs{i+1} = imrotate(imgs{i+1},rotate_angle,'crop');
	end
	imageSizes(i,:) = size(imgs{1},1:2);
end


% 2 Detect SURF points


if Fig
	scrnsze = get(0,'MonitorPositions');
	axisargs = {'tickdir','out','xcolor','w','ycolor','w','color','none','yaxislocation','right','box','on'};
	figure('position',scrnsze(1,:),'color','k')
	drawnow
	scrnsze = get(gcf,'position');
	w = scrnsze(3) * 0.33;
	h = w * (size(imgs{1},1) / size(imgs{1},2));
	b = scrnsze(4) * 0.05;
end

if Alt
	disp('Getting static match points')
else
	disp('Detecting SURF points')
end
for i = 1:numImages - 1
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

	disp(['Image pair: ' fnams{i} ' / ' fnams{i + 1}])
	if Tilt
		disp(['     Static SURF point number: ' num2str(matchedPoints1_static_culled{i}.Count) ])
		disp(['     Median movement distance (pixels): ' num2str(distance_x_med_s(i,1)) ' (x); ' ...
		num2str(distance_y_med_s(i,1)) ' (y); '])
	end
	
	if ~Alt
		disp(['     Moving SURF point number: ' num2str(matchedPoints1_moving_culled{i}.Count) ])
	end

	% Need at least 4 points
	if matchedPoints1_static_culled{i}.Count < 4 && Tilt
		warning('Too few SURF points for geometric transformation')
		continue
	else
		if Tilt
			% calculate transformation for image pairs
			rng('default')
			tforms(i + 1) = estgeotform2d(matchedPoints2_static_culled{i},matchedPoints1_static_culled{i},'projective');
			outputView = imref2d(size(imgs{i + 1}));
		end
		
		% Default method by transforming initial set of match points
		if ~Alt
			UV1{i} = [matchedPoints1_moving_culled{i}.Location(:,1), matchedPoints1_moving_culled{i}.Location(:,2)];
			XY2 = [matchedPoints2_moving_culled{i}.Location(:,1), matchedPoints2_moving_culled{i}.Location(:,2)];
			UV2{i} = transformPointsForward(tforms(i + 1),XY2);

			% translation distances for each image
			xdist(i) = round(prctile(UV1{i}(:,1) - UV2{i}(:,1),50));
			ydist(i) = round(prctile(UV1{i}(:,2) - UV2{i}(:,2),50));
			disp(['     ' fnams{i + 1} ' Distance from ' fnams{i} ': ' num2str(xdist(i)) ' (x), ' ...
			num2str(ydist(i)) ' (y) pixels'])
		end
	end
	if Tilt
		try
			imgs{i + 1} = imwarp(imgs{i + 1}, tforms(i + 1),'outputview',outputView);
		catch ME
			disp(ME)
			Alt = true;
			disp('Trying alternate method (extract new SURF points)')
		end
	end
	if Fig
		if exist('H1','var')
			delete(H1)
		end
		H1 = axes('units','pixels','position',[1, b, w, h]);
		image(imgs{i})
		hold(gca,'on')
		scatter(UV1{i}(:,1),UV1{i}(:,2))
		set(H1,axisargs{:},'xtick',[],'ytick',[],'ydir','reverse')
		title(strrep(fnams{i},'_','-'),'color','w')
		drawnow
		
		if exist('H2','var')
			delete(H2)
		end
		H2 = axes('units','pixels','position',[1 + w, b, w, h]);
		image(imgs{i + 1})
		set(H2,axisargs{:},'xtick',[],'ytick',[],'ydir','reverse')
		hold(gca,'on')
		scatter(UV2{i}(:,1),UV2{i}(:,2))
		title(strrep(fnams{i + 1},'_','-'),'color','w')
		drawnow
		
		if exist('H3','var')
			delete(H3)
		end
		H3 = axes('units','pixels','position',[11 + w * 2, b, w - 50, scrnsze(4) * 0.25]);
		stairs(2:i+1,xdist(1:i),'-ow')
		set(H3,axisargs{:},'xtick',2:numImages,'xlim',[2 numImages])
		title('Translation horizontal distance','color','w')
		ylabel('Pixels')
		xlabel('Image Number')
		drawnow

		if exist('H4','var')
			delete(H4)
		end
		H4 = axes('units','pixels','position',[11 + w * 2, 2 * b + scrnsze(4) * 0.25, w - 50, scrnsze(4) * 0.25]);
		stairs(2:i+1,ydist(1:i),'-ow')
		set(H4,axisargs{:},'xtick',2:numImages,'xlim',[2 numImages])
		title('Vertical adjustment distance','color','w')
		ylabel('Pixels')
		drawnow

		if exist('H5','var')
			delete(H5)
		end
		H5 = axes('units','pixels','position',[11 + w * 2, 3 * b + 2 * (scrnsze(4) * 0.25), w - 50, scrnsze(4) * 0.25]);
		hold(gca,'on')
		stairs(RGB(1:i,1),'-or')
		stairs(RGB(1:i,2),'-og')
		stairs(RGB(1:i,3),'-oc')
		set(H5,axisargs{:},'xtick',1:numImages,'xlim',[1 numImages])
		title('Chart color values','color','w')
		ylabel('Color')
		drawnow
		
		if i == numImages - 1
			delete(H5)
			H5 = axes('units','pixels','position',[11 + w * 2, 3 * b + 2 * (scrnsze(4) * 0.25), w - 50, scrnsze(4) * 0.25]);
			hold(gca,'on')
			stairs(RGB(:,1),'-or')
			stairs(RGB(:,2),'-og')
			stairs(RGB(:,3),'-oc')
			set(H5,axisargs{:},'xtick',1:numImages,'xlim',[1 numImages])
			title('Chart color values','color','w')
			ylabel('Color')
			drawnow
		end
	end
end


% 3. Calculate translation distances and build panorama


if Alt
	% New set of SURF points instead of transforming original points
	disp('Calculating translation distances')
	for i = 1:numel(imgs) - 1
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
		
		% translation distances for each image
		xdist(i) = round(prctile(matchedPoints1_moving_culled{i}.Location(:,1) - matchedPoints2_moving_culled{i}.Location(:,1),50));
		ydist(i) = round(prctile(matchedPoints1_moving_culled{i}.Location(:,2) - matchedPoints2_moving_culled{i}.Location(:,2),50));
		disp([fnams{i + 1}(foldern + 2:end) ' distance from ' fnams{i}(foldern + 2:end) ': ' num2str(xdist(i)) ' (x), ' ...
		num2str(ydist(i)) ' (y) pixels'])
	end
end

% Do the actual color correction after detecing all SURF points for reproducibility
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

% number of pixels on left to skip plus one (add a check for overlap using match points)
transpixels = 1500  + 1;

% initialzied matrix of ones, so position one = 1
xpos(1,2) = size(imgs{1},2);
% set 2:end start positions
xpos(2:end,1) = transpose(cumsum(xdist));
for i = 2:numel(imgs)
	xpos(i,2) = size(imgs{i },2) + xpos(i,1);
end

% transparent pixels will be excluded in the panorama
xpos(2:end,1) = xpos(2:end,1) + transpixels;

% length of panorama
xsize = xpos(end,2);

% initialize white panorma
if isa(imgs{1},'uint8')
	P = uint8(255 * ones(ysize,xsize,3));
else
	P = uint16(65535 * ones(ysize,xsize,3));
end

% first image which lacks transparency
P(ypos(1,1):ypos(1,2),xpos(1,1):xpos(1,2),:) = imgs{1};

% remaining images
for i = 2:numel(imgs)
	P(ypos(i,1):ypos(i,2),xpos(i,1):xpos(i,2),:) = imgs{i}(:,transpixels:end,:);
end

if Fig
	figure
	imshow(P)
end

toc