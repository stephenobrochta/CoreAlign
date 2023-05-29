function [angles_mode_m, angles_mode_s, distance_mode_m, distance_mode_s, ...
matchedPoints1_moving_culled, matchedPoints2_moving_culled, ...
matchedPoints1_static_culled, matchedPoints2_static_culled, ...
distances_moving, distances_static ...
] = removeOutlyingAngles(matchedPoints1_moving,matchedPoints2_moving, ...
matchedPoints1_static,matchedPoints2_static)

% calculate angles
x1_m = matchedPoints1_moving.Location(:,1);
y1_m = matchedPoints1_moving.Location(:,2);
x2_m = matchedPoints2_moving.Location(:,1);
y2_m = matchedPoints2_moving.Location(:,2);

x1_s = matchedPoints1_static.Location(:,1);
y1_s = matchedPoints1_static.Location(:,2);
x2_s = matchedPoints2_static.Location(:,1);
y2_s = matchedPoints2_static.Location(:,2);

angles_m = atan2d(y2_m - y1_m, x2_m - x1_m);
index = angles_m < 0;
angles_m(index) = angles_m(index) + 360;
angles_s = atan2d(y2_s - y1_s, x2_s - x1_s);
index = angles_s < 0;
angles_s(index) = angles_s(index) + 360;

% 50th percentile and 2 sig high and low
angles_mode_m = prctile(angles_m,[50, 100*(1-erf(2/sqrt(2)))/2, 100-100*(1-erf(2/sqrt(2)))/2]);
angles_mode_s = prctile(angles_s,[50, 100*(1-erf(2/sqrt(2)))/2, 100-100*(1-erf(2/sqrt(2)))/2]);

% cull out matched points that are outside of 2 sigma range
index = angles_m > angles_mode_m(2) & angles_m < angles_mode_m(3);

matchedPoints1_moving_culled = matchedPoints1_moving(index);
matchedPoints2_moving_culled = matchedPoints2_moving(index);
angles_m_culled = angles_m(index);

% cull out matched points that are outside of 2 sigma range
index = angles_s > angles_mode_s(2) & angles_s < angles_mode_s(3);

matchedPoints1_static_culled = matchedPoints1_static(index);
matchedPoints2_static_culled = matchedPoints2_static(index);
angles_s_culled = angles_s(index);

% recalculate precentiles.
angles_mode_m = prctile(angles_m_culled,[50, 100*(1-erf(2/sqrt(2)))/2, 100-100*(1-erf(2/sqrt(2)))/2]);
angles_mode_s = prctile(angles_s_culled,[50, 100*(1-erf(2/sqrt(2)))/2, 100-100*(1-erf(2/sqrt(2)))/2]);

distances_moving = diag(pdist2(matchedPoints1_moving_culled.Location,matchedPoints2_moving_culled.Location));
distances_static = diag(pdist2(matchedPoints1_static_culled.Location,matchedPoints2_static_culled.Location));

distance_mode_m = prctile(distances_moving,[50, 100*(1-erf(2/sqrt(2)))/2, 100-100*(1-erf(2/sqrt(2)))/2]);
distance_mode_s = prctile(distances_static,[50, 100*(1-erf(2/sqrt(2)))/2, 100-100*(1-erf(2/sqrt(2)))/2]);

