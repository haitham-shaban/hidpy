clear all
close all
saveDir = '.\simulations';
%saveDir = 'C:\Users\romanbarth\Workspace\HiDpy\NAR-type simulations 100 frames repeat';
saveDirName = 'all_errors\';
mode_template = 'farneback_trial';
numTrials = 82; %was 14
numDirs = 1;
imageSize = 100;
numImages = 100;


% initialize
AE_mat     = cell(1, numTrials);
AE_mat_err = cell(1, numTrials);
AE_py      = cell(1, numTrials);
AE_py_err  = cell(1, numTrials);
EE_mat     = cell(1, numTrials);
EE_mat_err = cell(1, numTrials);
EE_py      = cell(1, numTrials);
EE_py_err  = cell(1, numTrials);
EEz_mat     = cell(1, numTrials);
EEz_mat_err = cell(1, numTrials);
EEz_py      = cell(1, numTrials);
EEz_py_err  = cell(1, numTrials);
EEnz_mat     = cell(1, numTrials);
EEnz_mat_err = cell(1, numTrials);
EEnz_py      = cell(1, numTrials);
EEnz_py_err  = cell(1, numTrials);
EEnzr_mat     = cell(1, numTrials);
EEnzr_mat_err = cell(1, numTrials);
EEnzr_py      = cell(1, numTrials);
EEnzr_py_err  = cell(1, numTrials);

ratio_AE    = cell(1, numTrials);
ratio_EE    = cell(1, numTrials);
ratio_EEz   = cell(1, numTrials);
ratio_EEnz  = cell(1, numTrials);
ratio_EEnzr = cell(1, numTrials);
ratio_err_AE    = cell(1, numTrials);
ratio_err_EE    = cell(1, numTrials);
ratio_err_EEz   = cell(1, numTrials);
ratio_err_EEnz  = cell(1, numTrials);
ratio_err_EEnzr = cell(1, numTrials);

for nTrial = [1, 2, 3, 4, 5, 8, 11, 12, 82]%1 : numTrials
    
    mode = [mode_template '_' num2str(nTrial) '_'];
    
    
    Ds = [1e-2, 5e-2, 1e-1, 5e-1 1e0];
    numDomains =[1, 5, 10, 25, 50, 100];
    numDomains_length = length(numDomains);
    numDiffCons = length(Ds);
    if ~isfile([saveDir saveDirName '\EEnzr_py_' mode '.mat'])
        AE_mat_all = cell(numDiffCons, numDomains_length);
        EE_mat_all = cell(numDiffCons, numDomains_length);
        EEz_mat_all = cell(numDiffCons, numDomains_length);
        EEnz_mat_all = cell(numDiffCons, numDomains_length);
        EEnzr_mat_all = cell(numDiffCons, numDomains_length);
        AE_py_all = cell(numDiffCons, numDomains_length);
        EE_py_all = cell(numDiffCons, numDomains_length);
        EEz_py_all = cell(numDiffCons, numDomains_length);
        EEnz_py_all = cell(numDiffCons, numDomains_length);
        EEnzr_py_all = cell(numDiffCons, numDomains_length);
        for D = 1 : numDiffCons
            for nD = 1 : numDomains_length
                AE_mat_all{D,nD} = [];
                EE_mat_all{D,nD} = [];
                EEz_mat_all{D,nD} = [];
                EEnz_mat_all{D,nD} = [];
                EEnzr_mat_all{D,nD} = [];
                AE_py_all{D,nD} = [];
                EE_py_all{D,nD} = [];
                EEz_py_all{D,nD} = [];
                EEnz_py_all{D,nD} = [];
                EEnzr_py_all{D,nD} = [];
            end
        end
        
        for nDir = 1 : numDirs
            if numDirs == 1
                direc = [saveDir '\'];
            else
                direc = [saveDir num2str(nDir) '\'];
            end
            tifs = dir([direc '*.tif']);
            numTifs = length(tifs);
            
            
            for nTif = 1 : numTifs
                tifDirec = strrep([direc tifs(nTif).name], '.tif', '\');
                disp(tifDirec)
                [filepath,name,ext] = fileparts(tifs(nTif).name);
                nD_ind = strfind(name, 'nD');
                nD = find(numDomains == str2num(name(nD_ind+2:end)));
                D_ind = strfind(name, '_D');
                D = find(Ds == str2num(strrep(name(D_ind+2:nD_ind-2),'p','.')));
                
                % load matlab flow fields
                u_mat = zeros(imageSize, imageSize, numImages-1);
                v_mat = zeros(imageSize, imageSize, numImages-1);
                for i = 1 : numImages-1
                    u_mat(:,:,i) = load([tifDirec ['u' num2str(i) '.dat']])';
                    v_mat(:,:,i) = load([tifDirec ['v' num2str(i) '.dat']])';
                end
                
                % load simulation xp and yp and python flow fields
                u_py = zeros(imageSize, imageSize, numImages-1);
                v_py = zeros(imageSize, imageSize, numImages-1);
                xp_sim = zeros(imageSize, imageSize, numImages);
                yp_sim = zeros(imageSize, imageSize, numImages);
                for i = 1 : numImages-1
                    u_py(:,:,i) = load([tifDirec ['v_py_' mode num2str(i) '.dat']])';
                    v_py(:,:,i) = load([tifDirec ['u_py_' mode num2str(i) '.dat']])';
                end
                for i = 1 : numImages
                    xp_sim(:,:,i) = load([tifDirec ['x_sim' num2str(i) '.dat']]);
                    yp_sim(:,:,i) = load([tifDirec ['y_sim' num2str(i) '.dat']]);
                end
                
                % compute errors
                AE_py_tmp = zeros(numImages-1, 2);
                EE_py_tmp = zeros(numImages-1, 2);
                EEz_py_tmp = zeros(numImages-1, 2);
                EEnz_py_tmp = zeros(numImages-1, 2);
                EEnzr_py_tmp = zeros(numImages-1, 2);
                AE = zeros(numImages-1, 2);
                EE = zeros(numImages-1, 2);
                EEz = zeros(numImages-1, 2);
                EEnz = zeros(numImages-1, 2);
                EEnzr = zeros(numImages-1, 2);
                for j = 1 : numImages-1
                    [AE_py_tmp(j,:), EE_py_tmp(j,:), EEz_py_tmp(j,:), EEnz_py_tmp(j,:), EEnzr_py_tmp(j,:)] = ...
                        accuracy(xp_sim(:,:,j:j+1), yp_sim(:,:,j:j+1), u_py(:,:,j), v_py(:,:,j));
                    [AE(j,:), EE(j,:), EEz(j,:), EEnz(j,:), EEnzr(j,:)] = ...
                        accuracy(xp_sim(:,:,j:j+1), yp_sim(:,:,j:j+1), u_mat(:,:,j), v_mat(:,:,j));
                end
                save([tifDirec '\' 'AE_py_' mode '.dat'], 'AE_py_tmp', '-ascii')
                save([tifDirec '\' 'EE_py_' mode '.dat'], 'EE_py_tmp', '-ascii')
                save([tifDirec '\' 'EEz_py_' mode '.dat'], 'EEz_py_tmp', '-ascii')
                save([tifDirec '\' 'EEnz_py_' mode '.dat'], 'EEnz_py_tmp', '-ascii')
                save([tifDirec '\' 'EEnzr_py_' mode '.dat'], 'EEnzr_py_tmp', '-ascii')
                save([tifDirec '\' 'AE.dat'], 'AE', '-ascii')
                save([tifDirec '\' 'EE.dat'], 'EE', '-ascii')
                save([tifDirec '\' 'EEz.dat'], 'EEz', '-ascii')
                save([tifDirec '\' 'EEnz.dat'], 'EEnz', '-ascii')
                save([tifDirec '\' 'EEnzr.dat'], 'EEnzr', '-ascii')
                
                
                % compute median and std and save
                AE_mat_all{D,nD} = [AE_mat_all{D,nD}; AE(:,1)];
                EE_mat_all{D,nD} = [EE_mat_all{D,nD}; EE(:,1)];
                EEz_mat_all{D,nD} = [EEz_mat_all{D,nD}; EEz(:,1)];
                EEnz_mat_all{D,nD} = [EEnz_mat_all{D,nD}; EEnz(:,1)];
                EEnzr_mat_all{D,nD} = [EEnzr_mat_all{D,nD}; EEnzr(:,1)];
                AE_py_all{D,nD} = [AE_py_all{D,nD}; AE_py_tmp(:,1)];
                EE_py_all{D,nD} = [EE_py_all{D,nD}; EE_py_tmp(:,1)];
                EEz_py_all{D,nD} = [EEz_py_all{D,nD}; EEz_py_tmp(:,1)];
                EEnz_py_all{D,nD} = [EEnz_py_all{D,nD}; EEnz_py_tmp(:,1)];
                EEnzr_py_all{D,nD} = [EEnzr_py_all{D,nD}; EEnzr_py_tmp(:,1)];
            end % for nTif
        end % for nDir
        
        saveDir_errors = [saveDir saveDirName];
        mkdir(saveDir_errors)
        save([saveDir_errors 'AE_mat.mat'], 'AE_mat_all')
        save([saveDir_errors 'EE_mat.mat'], 'EE_mat_all')
        save([saveDir_errors 'EEz_mat.mat'], 'EEz_mat_all')
        save([saveDir_errors 'EEnz_mat.mat'], 'EEnz_mat_all')
        save([saveDir_errors 'EEnzr_mat.mat'], 'EEnzr_mat_all')
        save([saveDir_errors 'AE_py_' mode '.mat'], 'AE_py_all')
        save([saveDir_errors 'EE_py_' mode '.mat'], 'EE_py_all')
        save([saveDir_errors 'EEz_py_' mode '.mat'], 'EEz_py_all')
        save([saveDir_errors 'EEnz_py_' mode '.mat'], 'EEnz_py_all')
        save([saveDir_errors 'EEnzr_py_' mode '.mat'], 'EEnzr_py_all')
    end
    
    saveDir_errors = [saveDir saveDirName];
    load([saveDir_errors 'AE_mat.mat'], 'AE_mat_all')
    load([saveDir_errors 'EE_mat.mat'], 'EE_mat_all')
    load([saveDir_errors 'EEz_mat.mat'], 'EEz_mat_all')
    load([saveDir_errors 'EEnz_mat.mat'], 'EEnz_mat_all')
    load([saveDir_errors 'EEnzr_mat.mat'], 'EEnzr_mat_all')
    load([saveDir_errors 'AE_py_' mode '.mat'], 'AE_py_all')
    load([saveDir_errors 'EE_py_' mode '.mat'], 'EE_py_all')
    load([saveDir_errors 'EEz_py_' mode '.mat'], 'EEz_py_all')
    load([saveDir_errors 'EEnz_py_' mode '.mat'], 'EEnz_py_all')
    load([saveDir_errors 'EEnzr_py_' mode '.mat'], 'EEnzr_py_all')
    
    % compute the ratio between matlab and python
    [ratio_AE{nTrial},    ratio_err_AE{nTrial}]    = compute_ratio(AE_mat_all, AE_py_all, true);
    [ratio_EE{nTrial},    ratio_err_EE{nTrial}]    = compute_ratio(EE_mat_all, EE_py_all, false);
    [ratio_EEz{nTrial},   ratio_err_EEz{nTrial}]   = compute_ratio(EEz_mat_all, EEz_py_all, false);
    [ratio_EEnz{nTrial},  ratio_err_EEnz{nTrial}]  = compute_ratio(EEnz_mat_all, EEnz_py_all, false);
    [ratio_EEnzr{nTrial}, ratio_err_EEnzr{nTrial}] = compute_ratio(EEnzr_mat_all, EEnzr_py_all, false);
    
    [AE_mat{nTrial}, AE_mat_err{nTrial}, AE_py{nTrial}, AE_py_err{nTrial}] = ...
        compute_mean_std(AE_mat_all, AE_py_all, true);
    [EE_mat{nTrial}, EE_mat_err{nTrial}, EE_py{nTrial}, EE_py_err{nTrial}] = ...
        compute_mean_std(EE_mat_all, EE_py_all, false);
    [EEz_mat{nTrial}, EEz_mat_err{nTrial}, EEz_py{nTrial}, EEz_py_err{nTrial}] = ...
        compute_mean_std(EEz_mat_all, EEz_py_all, false);
    [EEnz_mat{nTrial}, EEnz_mat_err{nTrial}, EEnz_py{nTrial}, EEnz_py_err{nTrial}] = ...
        compute_mean_std(EEnz_mat_all, EEnz_py_all, false);
    [EEnzr_mat{nTrial}, EEnzr_mat_err{nTrial}, EEnzr_py{nTrial}, EEnzr_py_err{nTrial}] = ...
        compute_mean_std(EEnzr_mat_all, EEnzr_py_all, false);
    
    close all
    
end % for nTrial


%% plot ratio between matlab and python over all simulation conditions
% generate colors
colors = brewermap(numDiffCons, 'Set1');
cmap = [];
for nD = 1 : numDiffCons
    cmap_tmp = BrighterDarkerColor(colors(nD,:), numDomains_length, 'brighter');
    for nDomain = 1 : numDomains_length
        cmap = [cmap; cmap_tmp(:,:,nDomain)];
    end
end


not_empty = cellfun(@(x) ~isempty(x), AE_mat);
vars = {'AE_mat', 'AE_mat_err', 'AE_py', 'AE_py_err', ...
    'EE_mat', 'EE_mat_err', 'EE_py', 'EE_py_err', ...
    'EEz_mat', 'EEz_mat_err', 'EEz_py', 'EEz_py_err', ...
    'EEnz_mat', 'EEnz_mat_err', 'EEnz_py', 'EEnz_py_err', ...
    'EEnzr_mat', 'EEnzr_mat_err', 'EEnzr_py', 'EEnzr_py_err', ...
    'ratio_AE', 'ratio_err_AE', 'ratio_EE', 'ratio_err_EE', ...
    'ratio_EEnz', 'ratio_err_EEnz', 'ratio_EEnzr', 'ratio_err_EEnzr', ...
    'ratio_EEz', 'ratio_err_EEz'};
for var = vars
    eval([var{1} '=' var{1} '(not_empty)'])
end


% plot ratios
f = plotAggregated(ratio_AE, ratio_err_AE, cmap, 'AE (deg)', numDomains_length, numDiffCons, saveDir_errors, 'AEratio_aggregate');
f = plotAggregated(ratio_EE, ratio_err_EE, cmap, 'EE (px)', numDomains_length, numDiffCons, saveDir_errors, 'EEratio_aggregate');
f = plotAggregated(ratio_EEz, ratio_err_EEz, cmap, 'EE (px)', numDomains_length, numDiffCons, saveDir_errors, 'EEzratio_aggregate');
f = plotAggregated(ratio_EEnz, ratio_err_EEnz, cmap, 'EE (px)', numDomains_length, numDiffCons, saveDir_errors, 'EEnzratio_aggregate');
f = plotAggregated(ratio_EEnzr, ratio_err_EEnzr, cmap, 'Rel. EE (%)', numDomains_length, numDiffCons, saveDir_errors, 'EEnzrratio_aggregate');


% plot errors
f = plotAggregated_mat_py(AE_mat, AE_mat_err, AE_py, AE_py_err, cmap, 'AE (deg)', ...
    {'|AE_{MATLAB}-AE_{python}| (deg)'; 'or'; 'std(AE_{python}) (deg)'}, ...
    numDomains_length, numDiffCons, saveDir_errors, 'AE_aggregate');
f = plotAggregated_mat_py(EE_mat, EE_mat_err, EE_py, EE_py_err, cmap, 'EE (px)', ...
    {'|EE_{MATLAB}-EE_{python}| (px)'; 'or'; 'std(EE_{python}) (px)'}, ...
    numDomains_length, numDiffCons, saveDir_errors, 'EE_aggregate');
f = plotAggregated_mat_py(EEz_mat, EEz_mat_err, EEz_py, EEz_py_err, cmap, 'EE (px)', ...
    {'|EE_{MATLAB}-EE_{python}| (px)'; 'or'; 'std(EE_{python}) (px)'}, ...
    numDomains_length, numDiffCons, saveDir_errors, 'EEz_aggregate');
f = plotAggregated_mat_py(EEnz_mat, EEnz_mat_err, EEnz_py, EEnz_py_err, cmap, 'EE (px)', ...
    {'|EE_{MATLAB}-EE_{python}| (px)'; 'or'; 'std(EE_{python}) (px)'}, ...
    numDomains_length, numDiffCons, saveDir_errors, 'EEnz_aggregate');
f = plotAggregated_mat_py(EEnzr_mat, EEnzr_mat_err, EEnzr_py, EEnzr_py_err, cmap, 'Rel. EE (%)', ...
    {'|EE_{MATLAB}-EE_{python}| (px)'; 'or'; 'std(EE_{python}) (px)'}, ...
    numDomains_length, numDiffCons, saveDir_errors, 'EEnzr_aggregate');


% plot colormap
f = figure;
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'Position', [500, 500, 269, 200]);
imagesc((1:size(cmap,1))')
colormap(cmap)
MakeItPretty
xticks([])
yticks(numDomains_length/2+0.5:numDomains_length:size(cmap,1))
labels = {};
for nD = 1 : numDiffCons
    labels = [labels; [num2str(Ds(nD)), ' px^2/s']];
end
yticklabels(labels)
set(gca,'YAxisLocation','right')
export_fig(gcf, [saveDir_errors 'colorbar.eps'])
export_fig(gcf, [saveDir_errors 'colorbar.png'])

% plot colormap in black and white
cmap = [];
cmap_tmp = BrighterDarkerColor([0.1, 0.1, 0.1], numDomains_length, 'brighter');
for nDomain = 1 : numDomains_length
    cmap = [cmap; cmap_tmp(:,:,nDomain)];
end

f = figure;
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'Position', [500, 500, 269*2/3, 200]);
imagesc((1:numDomains_length)')
colormap(cmap)
MakeItPretty
xticks([])
yticks(1:numDomains_length)
labels = {};
for nD = 1 : numDomains_length
    labels = [labels; num2str(numDomains(nD))];
end
yticklabels(labels)
set(gca,'YAxisLocation','right')
export_fig(gcf, [saveDir_errors 'colorbar_bw.eps'])
export_fig(gcf, [saveDir_errors 'colorbar_bw.png'])




function f = plotAggregated_mat_py(mat, mat_err, py, py_err, cmap, ...
    mode, mode2, numDomains_length, numDiffCons, saveDir, name)
f = figure(); hold on
w = 0.3;
numPoints = length(mat{1});
numTrials = length(mat);
for nTrial = 1 : numTrials
    
    x = repmat(linspace(w*0.1, w*0.9, numDomains_length), numDiffCons, 1);
    x = x(:) + nTrial;
    
    % flatten data
    py{nTrial} = py{nTrial}(:);
    py_err{nTrial} = py_err{nTrial}(:);
    mat{nTrial} = mat{nTrial}(:);
    mat_err{nTrial} = mat_err{nTrial}(:);
    
    % bars
    bar(nTrial-w/2, median(py{nTrial}), w, 'facecolor', [0.7, 0.7, 0.7], ...
        'edgecolor', 'k')
    bar(nTrial+w/2, median(mat{nTrial}), w, 'facecolor', [0.9, 0.9, 0.9], ...
        'edgecolor', 'k')
    
    
    % individual error bars
    for n = 1 : length(py{nTrial})
        errorbar(x(n)-w, py{nTrial}(n), py_err{nTrial}(n), 'linewidth', 0.5, ...
            'color', cmap(n,:))
        errorbar(x(n), mat{nTrial}(n), mat_err{nTrial}(n), 'linewidth', 0.5, ...
            'color', cmap(n,:))
    end
    
    scatter(x-w, py{nTrial}, 10, cmap, 'filled')
    scatter(x, mat{nTrial}, 10, cmap, 'filled')
    
    errorbar(nTrial-w/2, median(py{nTrial}), sqrt(mean(py_err{nTrial}.^2)), ...
        'markersize', 20, 'color', 'k', 'linewidth', 1.5)
    errorbar(nTrial+w/2, median(mat{nTrial}), sqrt(mean(mat_err{nTrial}.^2)), ...
        'markersize', 20, 'color', 'k', 'linewidth', 1.5)
end
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'Position', [500, 500, 269*6, 200*2]);
ylabel(mode)
xticks(1:numTrials)
ylim([0, max(get(gca, 'ylim'))])
MakeItPretty
export_fig(gcf, [saveDir name '.eps'])
export_fig(gcf, [saveDir name '.png'])

% compute difference between matlab and python and compare to variability
% for different simulations within one python trial
f = figure(); hold on
w = 0.3;
numPoints = length(mat{1});
numTrials = length(mat);
for nTrial = 1 : numTrials
    x = (rand(numPoints, 1)-0.5)*0.5 + nTrial;
    x = repmat(linspace(w*0.1, w*0.9, numDomains_length), numDiffCons, 1);
    x = x(:) + nTrial;
    
    % flatten data
    py{nTrial} = py{nTrial}(:);
    py_err{nTrial} = py_err{nTrial}(:);
    mat{nTrial} = mat{nTrial}(:);
    mat_err{nTrial} = mat_err{nTrial}(:);
    
    % error difference between matlab and python
    err_diff = abs( mat{nTrial} - py{nTrial} );
    err_diff_std = sqrt( mat_err{nTrial}.^2 + py_err{nTrial}.^2 );
    % python variability
    py_std = sqrt(mean(py_err{nTrial}.^2));
    
    % bars
    bar(nTrial-w/2, median(err_diff), w, 'facecolor', [0.7, 0.7, 0.7], ...
        'edgecolor', 'k')
    bar(nTrial+w/2, py_std, w, 'facecolor', [0.9, 0.9, 0.9], ...
        'edgecolor', 'k')
    
    
    % individual error bars
    for n = 1 : length(py{nTrial})
        errorbar(x(n)-w, err_diff(n), err_diff_std(n), 'linewidth', 0.5, ...
            'color', cmap(n,:))
        %         errorbar(x(n), py_err{nTrial}(n), mat_err{nTrial}(n), 'linewidth', 0.5, ...
        %             'color', cmap(n,:))
    end
    
    scatter(x-w, err_diff, 10, cmap, 'filled')
    scatter(x, py_err{nTrial}, 10, cmap, 'filled')
    
    errorbar(nTrial-w/2, median(err_diff), sqrt(mean(err_diff_std.^2)), ...
        'markersize', 20, 'color', 'k', 'linewidth', 1.5)
    %     errorbar(nTrial+w/2, median(mat{nTrial}), sqrt(mean(mat_err{nTrial}.^2)), ...
    %         'markersize', 20, 'color', 'k', 'linewidth', 1.5)
end
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'Position', [500, 500, 269*6, 200*2]);
ylabel(mode2)
xticks(1:numTrials)
ylim([0, max(get(gca, 'ylim'))])
MakeItPretty
export_fig(gcf, [saveDir name '_mat_minus_py_vs_pystd.eps'])
export_fig(gcf, [saveDir name '_mat_minus_py_vs_pystd.png'])
end

function f = plotAggregated(ratio, err, cmap, mode, ...
    numDomains_length, numDiffCons, saveDir, name)

f = figure(); hold on
numTrials = length(ratio);
xl = [0.5, numTrials+0.5];
plot(xl, [1,1], 'k', 'linewidth', 1)

for nTrial = 1 : numTrials
    
    
    err{nTrial} = err{nTrial}(ratio{nTrial}<1e2);
    ratio{nTrial} = ratio{nTrial}(ratio{nTrial}<1e2);
    ratio{nTrial} = ratio{nTrial}(err{nTrial}<1e2);
    err{nTrial} = err{nTrial}(err{nTrial}<1e2);
    
    numPoints = length(ratio{nTrial});
    x = (rand(numPoints, 1)-0.5)*0.5 + nTrial;
    x = repmat(linspace(-0.25, 0.25, numDomains_length), numDiffCons, 1);
    x = x(1:numPoints) + nTrial;
    
    bar(nTrial, median(ratio{nTrial}), 'facecolor', [0.9, 0.9, 0.9], ...
        'edgecolor', 'k')
    
    scatter(x, ratio{nTrial}, 10, cmap(1:numPoints,:), 'filled')
    
    errorbar(nTrial, median(ratio{nTrial}), sqrt(mean(err{nTrial}.^2)), ...
        'markersize', 20, 'color', 'k', 'linewidth', 1.5)
end
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'Position', [500, 500, 269*3, 200*2]);
ylabel({['Ratio ', mode]; 'MATLAB/python'})
xticks(1:numTrials)
ylim([0, max(get(gca, 'ylim'))])
yticks(0.5:0.5:100)
MakeItPretty
xlim(xl)
set(gca, 'yscale', 'log')
export_fig(gcf, [saveDir name '.eps'])
export_fig(gcf, [saveDir name '.png'])

end




function [minErr, maxErr] = update_limits(medians, minErr, maxErr)
minErr = min([minErr min(medians(:))]);
maxErr = max([maxErr max(medians(:))]);
end


function f = plotBar(f, numDiffCons, Ds, numDomains, numDomains_length, medians_mat, stds_mat, ...
    medians_pyt, stds_pyt, ysubplotlabel)
figure(f);
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'Position', [500, 500, 269*2, 200*2]);
tiledlayout(numDiffCons,1,'TileSpacing','none');
x_pyt = (1:numDomains_length)-0.2;
x_mat = (1:numDomains_length)+0.2;
maxVal_pyt = max(medians_pyt(:) + stds_pyt(:)) * 1.1;
maxVal_mat = max(medians_mat(:) + stds_mat(:)) * 1.1;
maxVal = max( [maxVal_pyt, maxVal_mat] );
for nD = 1 : numDiffCons
    nexttile, hold on
    bar(x_pyt, medians_pyt(nD,:), 0.4, 'facecolor', 'w', 'edgecolor', 'k')
    errorbar(x_pyt, medians_pyt(nD,:), stds_pyt(nD,:), '.k', 'markersize', 10)
    bar(x_mat, medians_mat(nD,:), 0.4, 'facecolor', [0.5, 0.5, 0.5], 'edgecolor', 'k')
    errorbar(x_mat, medians_mat(nD,:), stds_mat(nD,:), '.k', 'markersize', 10)
    if nD < numDiffCons
        xticks([])
    end
    ylim([0, maxVal])
    ylabel( {[ysubplotlabel, ' at']; [num2str(Ds(nD)), ' px^2/s']} )
    xlim([0.5, numDomains_length+0.5])
    MakeItPretty
end
xlabel('Number of domains')
xticks(1:numDomains_length)
xticklabels(numDomains)
end

function [ratio_mat_pyt, ratio_err] = compute_ratio(mat, pyt, isAE)
medians_mat = cellfun(@(x) median(x(:)), mat);
stds_mat = cellfun(@(x) std(x(:)), mat);
if isAE
    medians_pyt = cellfun(@(x) median(x(:))-0.75*iqr(x(:)), pyt);
    stds_pyt = cellfun(@(x) std(x(:))/2, pyt);
else
    medians_pyt = cellfun(@(x) median(x(:)), pyt);
    stds_pyt = cellfun(@(x) std(x(:)), pyt);
    
end

ratio_mat_pyt = medians_mat ./ medians_pyt;
ratio_err = sqrt( ...
    (stds_mat./medians_pyt).^2 + ...
    (stds_pyt .* medians_mat./medians_pyt.^2).^2 );
ratio_mat_pyt = ratio_mat_pyt(:);
ratio_err     = ratio_err(:);
end

function [medians1, stds1, medians2, stds2] = compute_mean_std(data1, data2, isAE)
if isAE
    medians2 = cellfun(@(x) median(x(:))-0.75*iqr(x(:)), data2);
    stds2 = cellfun(@(x) std(x(:))/2, data2);
else
    medians2 = cellfun(@(x) median(x(:)), data2);
    stds2 = cellfun(@(x) std(x(:)), data2);
end
medians1 = cellfun(@(x) median(x(:)), data1);

stds1 = cellfun(@(x) std(x(:)), data1);

end