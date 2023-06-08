saveDir = '.\simulations\';
mkdir(saveDir)
addpath(genpath('.\MATLAB utils'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Variable parameters (there are a lot)
% General parameters
Grid.sx = 150; % size of picture in pixels 150
Optics.frames = 5;%100; % number of frames;
Cam.acqspeed = 0.2; % time delay between images in seconds
Cam.pixelsize = 0.106e-6; % pixelsize in meter
Movement = {'brownian', 'confined', 'directed'};

% Fluophores
Fluo.density = 50;%50
Fluo.Peak = 255; % Intensity peak value
Fluo.SB = Inf ; % SNR in dB
Fluo.Ion = 80000; % maximum Intensity in counts (photons)
Fluo.Ton = 0.04; % on-time in seconds
Fluo.Toff = 0; % off-time in seconds
Fluo.Tbl = Inf; % Bleaching time
Fluo.borderPercentage = 5; % Border where no particles are seeded in percentage of image size

% Camera: noise
Cam.readout_noise = 1.6;
Cam.dark_current = 0.06;
Cam.quantum_efficiency = 0.7;
Cam.gain = 6;
Cam.thermal_noise = 6e-4;
Cam.quantum_gain = 4.2;

% Optics
Optics.NA = 1.42; % Numerical Aperture 1
Optics.NA2 = 1.49; % Numerical Aperture 2
Optics.wavelength = [0.3].*Optics.NA./0.61.*1e-6; % The vector is the FWHM of gaussian PSF in um %0.3
Optics.magnification = 1;

% Motion
Motion.D = 1e-2; % in px^2/s
Motion.springconstant = 3*sqrt(4*Motion.D*Cam.acqspeed); % in um
Motion.Vmean = 0.5*sqrt(4*Motion.D*Cam.acqspeed); % in um/s
steps = 5;
% Motion.numberOfdomains = 20;
Motion.SimMode = 'chaotic';
Motion.f = [2/3 1/3]; % f: fraction of each population (in domain 1 or 2)
Motion.k21 = 1; % k21: switch rate 2 -> 1
Motion.k12 = 1; % k12: switch rate 1 -> 2
Motion.MovMode = 'brownian'; % possible inputs: 'brownian', 'confined', 'directed'

% Domains
Fluo.DomainSim.diameter = 20; % diameter - diameter of domains in um
Fluo.DomainSim.concentration = 0; % concentration of domain 2 in domain 1 (0 - 1): concentration = AreaOfDomain2/(AreaOfDomain1 + AreaOfDomain2)
Fluo.DomainSim.variation = 5; % variation in domain size
Fluo.DomainSim.numberOfdomains = 20;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fixed parameters
defType = 'density';
Grid.sy = Grid.sx;
Grid.sz = 1;
Grid.sampling = 1;
Grid.blckSize = 1; % ?????
Fluo.duration = Optics.frames*Cam.acqspeed; % in seconds
genType = 'random'; % random distribution of fluophore position
Fluo.radius = 8e-9;
Grid.template_size = 7;
Grid.blckSize = 3;
Motion.dt = Fluo.duration/Optics.frames;
Motion.sig0 = 0;
Motion.range = Grid.sx;
Motion.randomStart = 1;
Motion.method = 'Domains'; % method: simulation method (1Particle1D, Switch, Domains)
Fluo.background = (Fluo.Peak/Fluo.SB)/Cam.gain;
Fluo.number = round((1-Fluo.borderPercentage/100)*(Cam.pixelsize*1e6)^2 * Grid.sx^2 * Fluo.density);
mask.msk = ones(Grid.sx);


D = [1e-2, 5e-2, 1e-1, 5e-1 1e0];
numDomains =[1, 5, 10, 25, 50, 100];
numSimulations = length(D) * length(numDomains);
nSimu = 0;
tic
for iD = 1 : length(D)
    for iNum = 1 : length(numDomains)
        
        nSimu = nSimu + 1;
        elapsedTime = toc;
        remainingTime = elapsedTime/nSimu * (numSimulations-nSimu);
        disp(['Simulation ' num2str(nSimu) ' of ' num2str(numSimulations) '. Elapsed time ' num2str(elapsedTime) ', remaining time ' num2str(remainingTime)]) 
       
        % look if the last file has been saved. if yes, skip
        savename = [saveDir 'im_D' strrep(num2str(D(iD)),'.','p') '_nD' strrep(num2str(numDomains(iNum)),'.','p') '.tif'];
        OFresultDir = strrep(savename, '.tif', '');
        lastSavedFile = [OFresultDir '\' 'EEnz' '.dat'];
        if isfile(lastSavedFile), continue, end
        
        Motion.D = D(iD); % in px^2/s
        Fluo.DomainSim.numberOfdomains = numDomains(iNum);


        [Grid, Optics, Cam, Fluo, Motion] = simImages(Grid, Optics, Cam, Fluo, Motion);
        stacks = simStacks(Fluo.emitters,Optics.frames,Optics,Cam,Fluo,Grid,1,0);
        im = zeros(100,100,Optics.frames);
        for i = 1 : Optics.frames
            im(:,:,i) = stacks.discrete(26:end-25, 26:end-25, i);
        end
        
        % compute xp and yp from the simulation
        [xp_sim, yp_sim] = meshgrid(1:Grid.sx);
        xp_sim = repmat(xp_sim,[1, 1, size(Fluo.emitters,3)]);
        yp_sim = repmat(yp_sim,[1, 1, size(Fluo.emitters,3)]);
        x = Fluo.emitters(:,1,1);
        y = Fluo.emitters(:,2,1);
        for j = 1:Optics.frames
            for k = 1:length(x)
                xp_sim(x(k),y(k),j) = Fluo.emitters(k,2,j);
                yp_sim(x(k),y(k),j) = Fluo.emitters(k,1,j);
            end
        end
        xp_sim = xp_sim(26:end-25, 26:end-25, :);
        yp_sim = yp_sim(26:end-25, 26:end-25, :);
        
        % save images
        im = im ./ max(im(:));
        im = im - min(im(:));
        im = uint8(im * 255);
        

        mkdir(OFresultDir);
        imwrite(im(:,:,1),savename)
        for i = 2 : Optics.frames
            imwrite(im(:,:,i),savename,'WriteMode','append')
        end
        
        % compute OF
        mask.msk = true(100);
        uout = cell(1,Optics.frames-1); vout = uout;
        for j = 1 : Optics.frames-1
%             progress('Optical Flow (classic)', j, length(im)-1, isunix)
            uv = estimate_flow_interface(im(:,:,j), im(:,:,j+1), mask.msk, 'classic+nl-fast');
            uout{j} = double(uv(:,:,1));
            vout{j} = double(uv(:,:,2));
        end

        [xp, yp] = integrationFORpool(uout, vout);

        for j=1:Optics.frames-1
            u = uout{j};
            v = vout{j};
            save([OFresultDir '\' 'u' num2str(j) '.dat'], 'u', '-ascii')
            save([OFresultDir, '\', 'v', num2str(j), '.dat'], 'v', '-ascii')
        end
        for j=1:Optics.frames
            x = xp(:,:,j);
            y = yp(:,:,j);
            save([OFresultDir '\' 'x' num2str(j) '.dat'], 'x', '-ascii')
            save([OFresultDir, '\', 'y', num2str(j), '.dat'], 'y', '-ascii')
        end
        for j=1:Optics.frames
            x = xp_sim(:,:,j);
            y = yp_sim(:,:,j);
            save([OFresultDir '\' 'x_sim' num2str(j) '.dat'], 'x', '-ascii')
            save([OFresultDir, '\', 'y_sim', num2str(j), '.dat'], 'y', '-ascii')
        end
        
        % compute errors
        AE = zeros(Optics.frames-1, 2);
        EE = zeros(Optics.frames-1, 2);
        EEz = zeros(Optics.frames-1, 2);
        EEnz = zeros(Optics.frames-1, 2);
        for j = 1 : Optics.frames-1
            [AE(j,:), EE(j,:), EEz(j,:), EEnz(j,:)] = ...
                accuracy(xp_sim(:,:,j:j+1), yp_sim(:,:,j:j+1), uout{j}, vout{j});
        end
        save([OFresultDir '\' 'AE' '.dat'], 'AE', '-ascii')
        save([OFresultDir '\' 'EE' '.dat'], 'EE', '-ascii')
        save([OFresultDir '\' 'EEz' '.dat'], 'EEz', '-ascii')
        save([OFresultDir '\' 'EEnz' '.dat'], 'EEnz', '-ascii')
    end
end

