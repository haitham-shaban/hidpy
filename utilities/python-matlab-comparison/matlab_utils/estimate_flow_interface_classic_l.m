function uvo = estimate_flow_interface_classic_l(im1, im2, mask, method, params)

%ESTIMATE_FLOW_INTERFACE  Optical flow estimation with various methods
%
% Demo program
%     [im1, im2, tu, tv] = read_flow_file('middle-other', 1);
%     uv = estimate_flow_interface(im1, im2, 'classic+nl-fast');
%     [aae stdae aepe] = flowAngErr(tu, tv, uv2(:,:,1), uv2(:,:,2), 0)
%
%
% Authors: Deqing Sun, Department of Computer Science, Brown University
% Contact: dqsun@cs.brown.edu
% $Date: $
% $Revision: $
%
% Copyright 2007-2010, Brown University, Providence, RI. USA
%
%                          All Rights Reserved
%
% All commercial use of this software, whether direct or indirect, is
% strictly prohibited including, without limitation, incorporation into in
% a commercial product, use in a commercial service, or production of other
% artifacts for commercial purposes.
%
% Permission to use, copy, modify, and distribute this software and its
% documentation for research purposes is hereby granted without fee,
% provided that the above copyright notice appears in all copies and that
% both that copyright notice and this permission notice appear in
% supporting documentation, and that the name of the author and Brown
% University not be used in advertising or publicity pertaining to
% distribution of the software without specific, written prior permission.
%
% For commercial uses contact the Technology Venture Office of Brown University
%
% THE AUTHOR AND BROWN UNIVERSITY DISCLAIM ALL WARRANTIES WITH REGARD TO
% THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
% FITNESS FOR ANY PARTICULAR PURPOSE.  IN NO EVENT SHALL THE AUTHOR OR
% BROWN UNIVERSITY BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL
% DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR
% PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
% ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
% THIS SOFTWARE.


% Read in arguments
method = 'classic-l'; % only this one is allowed here
if nargin < 3
    method = 'classic-l';
end;

if (~isdeployed)
    addpath(genpath('utils'));
end

% Load default parameters
median_filter_size = [5 5];
ope.lambda          = 1;
ope.lambda_q        = 1;    % Quadratic formulation of the objective function

ope.sor_max_iters   = 1e4;       % 100 seems sufficient

ope.limit_update    = true;      % limit the flow incrment to be less than 1 per linearization step
ope.display         = false;

ope.solver          = 'backslash';   % 'sor' 'pcg' for machines with limited moemory
ope.deriv_filter    = [1 -8 0 8 -1]/12; % 5-point 7 point [-1 9 -45 0 45 -9 1]/60;
ope.blend           = 0.5;           % temporal blending ratio

ope.texture         = false;     % use texture component as input
ope.fc              = false;     % use filter constancy


ope.interpolation_method = 'cubic';  % 'bi-cubic', 'cubic', 'bi-linear'

% For Graduated Non-Convexity (GNC) optimization
ope.gnc_iters       = 3;
ope.alpha           = 1;             % change linearly from 1 to 0 through the GNC stages

ope.max_iters       = 10;            % number of warping per pyramid level
ope.max_linear      = 1;             % maximum number of linearization performed per warping, 1 OK for HS

% For GNC stage 1
ope.pyramid_levels  = 4;
ope.pyramid_spacing = 2;

% For GNC stage 2 to last
ope.gnc_pyramid_levels     = 2;
ope.gnc_pyramid_spacing    = 1.25;

method = 'lorentzian'; %'geman_mcclure'; %charbonnier
ope.spatial_filters = {[1 -1], [1; -1]};
for i = 1:length(ope.spatial_filters);
    ope.rho_spatial_u{i}   = robust_function(method, 0.03);
    ope.rho_spatial_v{i}   = robust_function(method, 0.03);
end;
ope.rho_data        = robust_function(method, 1.5); % 6.3

ope.alp             = 0.95;  % for ROF texture decomposition

ope.color_images     = ones(1,1,3);
ope.auto_level       = true;
ope.median_filter_size   = median_filter_size;
ope.area_hsz = 7;
ope.sigma_i = 7;
ope.fullVersion = 0;

method = 'lorentzian'; %'geman_mcclure';
ope.spatial_filters = {[1 -1], [1; -1]};
for i = 1:length(ope.spatial_filters);
    ope.rho_spatial_u{i}   = robust_function(method, 0.1);
    ope.rho_spatial_v{i}   = robust_function(method, 0.1);
end;
ope.rho_data        = robust_function(method, 3.5);

ope.lambda = 0.045;
ope.lambda_q = 0.045 ;
ope.median_filter_size   = median_filter_size;
ope.texture              = true;
ope.alp = 0.95;

method = 'lorentzian'; %'geman_mcclure';
ope.spatial_filters = {[1 -1], [1; -1]};
for i = 1:length(ope.spatial_filters);
    ope.rho_spatial_u{i}   = robust_function(method, 0.03);
    ope.rho_spatial_v{i}   = robust_function(method, 0.03);
end;
ope.rho_data        = robust_function(method, 1.5);

ope.lambda = 0.06;
ope.lambda_q = 0.06 ;

if nargin > 4%%%%%%%%%%%%%%%%%%%%%%%%%%
    ope = parse_input_parameter(ope, params);
end;

% Uncomment this line if Error using ==> \  Out of memory. Type HELP MEMORY for your option.
%ope.solver    = 'pcg';

if size(im1, 3) > 1
    tmp1 = double(rgb2gray(uint8(im1)));
    tmp2 = double(rgb2gray(uint8(im2)));
    ope.images  = cat(length(size(tmp1))+1, tmp1, tmp2);
else
    
    if isinteger(im1);
        im1 = double(im1);
        im2 = double(im2);
    end;
    ope.images  = cat(length(size(im1))+1, im1, im2);
end;

% Use color for weighted non-local term
if ~isempty(ope.color_images)
    if size(im1, 3) > 1
        % Convert to Lab space
        im1 = RGB2Lab(im1);
        for j = 1:size(im1, 3);
            im1(:,:,j) = scale_image(im1(:,:,j), 0, 255);
        end;
    end;
    ope.color_images   = im1;
end;

% Compute flow field
uv  = compute_flow(mask, ope, zeros([size(im1,1) size(im1,2) 2]));

if nargout == 1
    uvo = uv;
end;
