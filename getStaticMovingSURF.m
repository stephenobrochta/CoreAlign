function [matchedPoints1_moving, ...
matchedPoints2_moving, ...
matchedPoints1_static, ...
matchedPoints2_static, ... 
stepping_moving, ...
stepping_static ...
] = getStaticMovingSURF(img1,img2,n,d)

% start with one pixel +/- the 50th percentile
[stepping_moving,stepping_static] = deal(1);

I1 = rgb2gray(img1);
I2 = rgb2gray(img2);


% find SURF features
points1 = detectSURFFeatures(I1);
points2 = detectSURFFeatures(I2);

% Extract SURF features
[f1,vpts1] = extractFeatures(I1,points1);
[f2,vpts2] = extractFeatures(I2,points2);

% Retreive locations of matched pairs
[indexPairs,~] = matchFeatures(f1,f2) ;
matchedPoints1 = vpts1(indexPairs(:,1));
matchedPoints2 = vpts2(indexPairs(:,2));

% optimze this
% There should be two distinct groups of points, those that are moving the same distance
% and those that are not moving. euclidean distances
distances = diag(pdist2(matchedPoints1.Location,matchedPoints2.Location));

% find the modes between 0 and 100 and 100 to max
index_static = distances <= 100;
index_moving = distances > 100;

% (probably) due to high precision in pixel locations (not whole numbers)
% there are often very few of the same value. so the mode may not resemble the histogram
mode_static = mode(round(distances(index_static)));
mode_moving = mode(round(distances(index_moving)));
clear index_moving index_static

% number to start with
index_moving = distances > (mode_moving - stepping_moving) & distances < (mode_moving + stepping_moving);
index_static = distances > (mode_static - stepping_static) & distances < (mode_static + stepping_static);

% If too few match points then increment allowable distance by one pixel
while length(find(index_moving)) < n
	stepping_moving = stepping_moving + 1;
	if stepping_moving == d, break, end
	index_moving = distances > (mode_moving - stepping_moving) & distances < (mode_moving + stepping_moving);
end

% get the points to use 
matchedPoints1_moving = vpts1(indexPairs(index_moving,1));
matchedPoints2_moving = vpts2(indexPairs(index_moving,2));
matchedPoints1_static = vpts1(indexPairs(index_static,1));
matchedPoints2_static = vpts2(indexPairs(index_static,2));

