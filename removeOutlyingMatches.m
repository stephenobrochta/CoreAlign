function [distance_x_m, distance_y_m, distance_x_s, distance_y_s, ...
matchedPoints1_moving_culled, matchedPoints2_moving_culled, ...
matchedPoints1_static_culled, matchedPoints2_static_culled, ...
distance_x_med_m, distance_y_med_m, distance_x_med_s, distance_y_med_s ...
] = removeOutlyingMatches(matchedPoints1_moving,matchedPoints2_moving, ...
matchedPoints1_static,matchedPoints2_static)

% extract x and y directions
% moving points
x1_m = matchedPoints1_moving.Location(:,1);
y1_m = matchedPoints1_moving.Location(:,2);
x2_m = matchedPoints2_moving.Location(:,1);
y2_m = matchedPoints2_moving.Location(:,2);

% static points
x1_s = matchedPoints1_static.Location(:,1);
y1_s = matchedPoints1_static.Location(:,2);
x2_s = matchedPoints2_static.Location(:,1);
y2_s = matchedPoints2_static.Location(:,2);

% calculate distances in x and y directions
distance_x_m = x1_m - x2_m;
distance_y_m = y1_m - y2_m;
distance_x_s = x1_s - x2_s;
distance_y_s = y1_s - y2_s;

% remove outlying x distances

% assuming the images are being take from the left to the right, moving distances should be positive
index = distance_x_m > 0;
distance_x_m = distance_x_m(index);
distance_y_m = distance_y_m(index);
matchedPoints1_moving_culled = matchedPoints1_moving(index);
matchedPoints2_moving_culled = matchedPoints2_moving(index);

% static distances can be in any direction so use 2 sigma
distance_x_med_s = prctile(distance_x_s,[50, 100*(1-erf(2/sqrt(2)))/2, 100-100*(1-erf(2/sqrt(2)))/2]);
index = distance_x_s > distance_x_med_s(2) & distance_x_s < distance_x_med_s(3);
distance_x_s = distance_x_s(index);
distance_y_s = distance_y_s(index);

% Geoslicer images have only a fraction of the color chart visible
% may not get enough matches for computing the transformation
if numel(find(index)) < 4
	matchedPoints1_static_culled = matchedPoints1_static;
	matchedPoints2_static_culled = matchedPoints2_static;
	warning([num2str(numel(find(index))) ' static SURF points remaining after culling. None removed'])
else
	matchedPoints1_static_culled = matchedPoints1_static(index);
	matchedPoints2_static_culled = matchedPoints2_static(index);
end

% remove outlying y directions

% y moving distances can also be an either direction
distance_y_med_m = prctile(distance_y_m,[50, 100*(1-erf(2/sqrt(2)))/2, 100-100*(1-erf(2/sqrt(2)))/2]);
index = distance_y_m > distance_y_med_m(2) & distance_y_m < distance_y_med_m(3);
distance_x_m = distance_x_m(index);
distance_y_m = distance_y_m(index);
matchedPoints1_moving_culled = matchedPoints1_moving(index);
matchedPoints2_moving_culled = matchedPoints2_moving(index);

% y static
distance_y_med_s = prctile(distance_y_s,[50, 100*(1-erf(2/sqrt(2)))/2, 100-100*(1-erf(2/sqrt(2)))/2]);
index = distance_y_s > distance_y_med_s(2) & distance_y_s < distance_y_med_s(3);
distance_x_s = distance_x_s(index);
distance_y_s = distance_y_s(index);
if numel(find(index)) < 4
	matchedPoints1_static_culled = matchedPoints1_static;
	matchedPoints2_static_culled = matchedPoints2_static;
	warning([num2str(numel(find(index))) ' static SURF points remaining after culling. None removed'])
else
	matchedPoints1_static_culled = matchedPoints1_static(index);
	matchedPoints2_static_culled = matchedPoints2_static(index);
end

% recalculate distances
distance_y_med_m = prctile(distance_y_m,[50, 100*(1-erf(2/sqrt(2)))/2, 100-100*(1-erf(2/sqrt(2)))/2]);
distance_x_med_m = prctile(distance_x_m,[50, 100*(1-erf(2/sqrt(2)))/2, 100-100*(1-erf(2/sqrt(2)))/2]);
distance_y_med_s = prctile(distance_y_s,[50, 100*(1-erf(2/sqrt(2)))/2, 100-100*(1-erf(2/sqrt(2)))/2]);
distance_x_med_s = prctile(distance_x_s,[50, 100*(1-erf(2/sqrt(2)))/2, 100-100*(1-erf(2/sqrt(2)))/2]);