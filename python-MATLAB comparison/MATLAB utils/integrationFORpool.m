function [xp, yp] = integrationFORpool(u, v)
% INTERGRATIONFORPOOL integrates the velocity fields u (and v) to get the
% trajectories stored as location information in xp and yp
%
%   INPUT
%   u, v:  cell arrays as given by im2velocityfield
%
%   OUTPUT
%   xp, yp: 3D matrix with size of u and length = length(u)+1. The matrices
%           consist of the position of every pixel for time t.
%
% written by Roman Barth, 15-05-2017

interpolation_method = 2;

if ~iscell(u) && ~iscell(v)
    u = mat3Dcell(u);
    v = mat3Dcell(v);
end

[xp(:,:,1), yp(:,:,1)] = meshgrid(1:size(u{1},2), 1:size(u{1},1));

x = xp(:,:,1); y = yp(:,:,1);

vx = zeros(size(u{1},1), size(u{1},2), length(u));
vy = zeros(size(u{1},1), size(u{1},2), length(u));


for j = 1:length(u)
    [u{j}, v{j}]=naninterp2(u{j},v{j},ones(size(u{j})),x,y);
    u{j}(isnan(u{j})) = 0;
    v{j}(isnan(v{j})) = 0;
    
    vx(:,:,j) = u{j};
    vy(:,:,j) = v{j};
end

% get mask
mask = sum(vx,3)==0 & sum(vy,3)==0;

% set 0-values to NaN which can then be interpolated later on
vx(vx==0) = NaN;
vy(vy==0) = NaN;


time = (1:length(u))';


[xx,yy] = ndgrid(1:size(u{1},2), 1:size(u{1},1));


warning('off')
for t = 1:length(time)
    if verLessThan('matlab', '8.1') % version 8.1 is R2013a
        vx_int = griddedInterpolant(xx, yy, vx(:,:,t)', 'pchip');
        vy_int = griddedInterpolant(xx, yy, vy(:,:,t)', 'pchip');
    else
        vx_int = griddedInterpolant(xx, yy, vx(:,:,t)', 'pchip', 'none');
        vy_int = griddedInterpolant(xx, yy, vy(:,:,t)', 'pchip', 'none');
        
    end
    
    
    xp(:,:,t+1) = xp(:,:,t) + vx_int(xp(:,:,t), yp(:,:,t));
    yp(:,:,t+1) = yp(:,:,t) + vy_int(xp(:,:,t), yp(:,:,t));
    
    % in case of NaN-values, interpolate them
    %     disp('interpolate')
    %     [xp(:,:,t+1), yp(:,:,t+1)]=naninterp2(xp(:,:,t+1),yp(:,:,t+1),ones(size(xp(:,:,t+1))),x,y);
    
    % get current trajectory positions
    xptemp = xp(:,:,t+1);
    % outside nucleus, trajectories do not move. Thus set initial coordinates
    % there
    xptemp(mask) = x(mask);
    % If there are still NaN's, interpolate them
    xp_t_plus_1 = inpaint_nans(xptemp, interpolation_method);
    % set values outside nucleus back to initial coordinates
    xp_t_plus_1(mask) = x(mask);
    % write to array
    xp(:,:,t+1) = xp_t_plus_1;
    
    % ... do the same for the y-component
    yptemp = yp(:,:,t+1);
    yptemp(mask) = y(mask);
    yp_t_plus_1 = inpaint_nans(yptemp, interpolation_method);
    yp_t_plus_1(mask) = y(mask);
    yp(:,:,t+1) = yp_t_plus_1;
end