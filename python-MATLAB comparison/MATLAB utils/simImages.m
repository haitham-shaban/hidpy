function [Grid, Optics, Cam, Fluo, Motion] = simImages(Grid, Optics, Cam, Fluo, Motion)

%% Simulate Domains for Density
Fluo.DomainSim   = domains_chaotic(Grid, Fluo.DomainSim);

%% Simulate positions for all images
[sim_pos, ~] =  simCoord(Fluo.number, Optics.frames, Cam.acqspeed, Motion.D, Motion.f, Motion.k21, Motion.k12,...
    Motion.range, Motion.randomStart, Motion.method, Motion.SimMode, Fluo.DomainSim.domains, Fluo.borderPercentage, Cam, Motion, Grid);

Fluo.emitters = zeros(Fluo.number,2,Optics.frames);
for j = 1:Optics.frames
    Fluo.emitters(:,:,j) = sim_pos(j:Optics.frames:Fluo.number*Optics.frames,5:6);
end

%% Create PSF
[Optics.psf,Optics.psf_digital,Optics.fwhm,Optics.fwhm_digital] = ...
    gaussianPSF(Optics.NA,Optics.magnification,Optics.wavelength,Fluo.radius,Cam.pixelsize); % Point-spread function of the optical system


%% Give brightness and process image (PSF, noise)
% im = cell(1,Optics.frames);

% stacks = simStacks(Fluo.emitters,Optics.frames,Optics,Cam,Fluo,Grid,1,0);
% for j = 1:Optics.frames
%     im{j} = stacks.discrete(:,:,j);
% end


% [xp, yp] = meshgrid(1:Grid.sx);
% 
% xp = repmat(xp,[1, 1, size(Fluo.emitters,3)]);
% yp = repmat(yp,[1, 1, size(Fluo.emitters,3)]);
% 
% 
% x = Fluo.emitters(:,1,1);
% y = Fluo.emitters(:,2,1);
% 
% for j = 1:Optics.frames
%     for k = 1:length(x)
%         xp(x(k),y(k),j) = Fluo.emitters(k,2,j);
%         yp(x(k),y(k),j) = Fluo.emitters(k,1,j);
%     end
% end
end