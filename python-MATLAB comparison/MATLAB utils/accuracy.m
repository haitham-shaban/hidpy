function [AE, EE, EEz, EEnz, EEnzr] = accuracy(xp, yp, u, v)

% Calculates the Angular error (AE) and endpoint error (EE) of the computed
% vector field [u v] in respect to the simulated [usim vsim].
%
% INPUT:
% usim, vsim, u, v are row or column cell arrays with the 2D
% velocity fields
%
% OUTPUT:
% AE:    vector of size u containing absolute difference of angles between
%        fields in degrees. First row contains values, second row contains
%        the standard deviation
% EE:    vector of size u containing relative difference in magnitude in
%        percent
% EEz:   vector of size u containing absolute difference in magnitude in
%        pixels for pixels which have zero magnitude in simulated images
% EEnz:  vector of size u containing difference in magnitude for
%        pixels which have non-zero magnitude in simulated field in percent
% EEnzr: vector of size u containing relative difference in magnitude for
%        pixels which have non-zero magnitude in simulated field in percent

% Author: Roman Barth, LBME (CNRS), Team Bystricky, 10/03/2017

% AE = zeros(2, length(u)-1); EE = AE; EEz = AE; EEnz = AE;

usim = xp(:,:,2)-xp(:,:,1);
vsim = yp(:,:,2)-yp(:,:,1);

iszero = xp(:,:,2)-xp(:,:,1) == 0;
% iszero = repmat(iszero, [1 1 size(xp,3)-1]);


%% Angluar error
% t = acos((ones(size(diff(xp,[],3))) + diff(xp,[],3).*diff(xp2,[],3) + diff(yp,[],3).*diff(yp2,[],3)) ./ ...
%     (sqrt(ones(size(diff(xp,[],3))) + diff(xp,[],3).*diff(xp,[],3) + diff(yp,[],3).*diff(yp,[],3)) .* ...
%     sqrt(ones(size(diff(xp,[],3))) + diff(xp2,[],3).*diff(xp2,[],3) + diff(yp2,[],3).*diff(yp2,[],3)))) .* 180/pi;

% t = acos((usim.*u+vsim.*v+1)./sqrt(1+usim.^2+vsim.^2)./sqrt(1+u.^2+v.^2)) * 180/pi;
t = acos((usim.*u+vsim.*v)./sqrt(usim.^2+vsim.^2)./sqrt(u.^2+v.^2)) * 180/pi;

t(~isfinite(t)) = NaN;

AE = zeros(2, size(t,3));
for j = 1:size(t,3)
    temp = t(:,:,j);
    AE(1,j) = nanmean(nanmean(abs(temp(~iszero))));
    AE(2,j) = nanstd(abs(temp(~iszero)));
end
% [AE(1,1), AE(2,1), ~]=flowAngErr(usim, vsim, u, v, 0);



%% Endpoint error
t = sqrt((u-usim).^2 + (v-vsim).^2);
mag_sim = sqrt((usim).^2 + (vsim).^2);
%     t = sqrt((u - usim).^2 + (v - vsim).^2);

% for zero elements, compute absolute difference (because x/0=Inf)
EEz = zeros(2, size(t,3));
for j = 1:size(t,3)
    temp = t(:,:,j);
    EEz(1,j) = nanmean(nanmean(abs(temp(iszero))));
    EEz(2,j) = nanstd(abs(temp(iszero)));
end

% get relative error by normalization in percent
%     t = abs( t ./ sqrt((usim).^2 + (vsim).^2));
%     comp = double(sqrt(u.^2+v.^2)>sqrt(usim.^2+vsim.^2));
%     comp(iszero) = NaN;
%     isbigger = nansum(comp(:))>sum(comp==0);
%     if ~isbigger 
%         t = -t;
%     end

t(~isfinite(t)) = NaN;
EE = zeros(2, size(t,3));
EEnz = zeros(2, size(t,3));
EEnzr = zeros(2, size(t,3));
for j = 1:size(t,3)
    temp = t(:,:,j);
    EE(1,j) = nanmean(nanmean(abs(temp)));
    EE(2,j) = nanstd(abs(temp(:)));
    
    EEnz(1,j) = nanmean(nanmean(abs(temp(~iszero))));
    EEnz(2,j) = nanstd(abs(temp(~iszero)));
    
    EEnzr(1,j) = nanmean(nanmean(abs(temp(~iszero)./mag_sim(~iszero))));
    EEnzr(2,j) = nanstd(abs(temp(~iszero)./mag_sim(~iszero)));
end
EEnzr = EEnzr * 100; % in percent

end