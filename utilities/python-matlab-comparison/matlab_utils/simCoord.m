function [sim_coordinates, stats] =  simCoord(particleCount, N, tau, D, f, k21, k12, range, randomStart, method, simMode, domains, borderPercentage, Cam, Motion, Grid)

% particleCount: number of particles to simulate
% N: number of steps for each particle
% tau: time interval in seconds
% D: Diffusion coefficients (um^2/s)
% f: fraction
% k21: switch rate 2 -> 1
% k12: switch rate 1 -> 2
% sig0: localization uncertainty
% range: simulation range (um)
% randomStart: random particle seed
% method: simulation method (1Particle1D, Switch, Domains)
% domains: square map of domains (0, 1)

%% prepare variables and matrices

k(1) = sqrt(D(1) * 4 * tau);  % displacements scaling factor for D1
D(2) = D(1);
k(2) = k(1);
% if D(2)~=0
%     k(2) = sqrt(D(2) * 2 * tau);  % displacements scaling factor for D2
% end
borderPercentage = borderPercentage/100;

% DATA STRUCTURE: 
% 1- cumulative 2-dataset 3- track# 4- frame# 5- X(um) 6- Y(um)
sim_coordinates = zeros(N*particleCount, 6); %allocate array
sim_coordinates(:, 2) = 1; %one dataset if data are simulated

switch method
    case '1particle1D'
   
    if numel(k) == 1 % if no 2nd Dcoeff is given set fraction value to 1
        f=1;
    end
    
    % fraction 1    
    for i = 1:round(particleCount*(f))
        particle(i).coordinates = zeros(N, 2);
        particle(i).coordinates(:, 1) = cumsum(k(1)*randn(1, N));
        particle(i).coordinates(:, 2) = cumsum(k(1)*randn(1, N));
        % waitbar(i/particleCount);  
    end

    % fraction 2
    if length(k) ~=1 
    for i = round(particleCount*(f))+1:particleCount
        particle(i).coordinates = zeros(N, 2);
        particle(i).coordinates(:, 1) = cumsum(k(2)*randn(1, N));
        particle(i).coordinates(:, 2) = cumsum(k(2)*randn(1, N));
        % waitbar((particleCount*(f)+i)/particleCount);  
    end
    end
    
    if randomStart % seed particles into 'range' sized square
        for i=1:particleCount
            particle(i).coordinates(:, 1) = particle(i).coordinates(:, 1)+random('unif', -range/4, range/4);
            particle(i).coordinates(:, 2) = particle(i).coordinates(:, 2)+random('unif', -range/4, range/4);
        end
    end    

    stats.actualF1 = round(particleCount*f)/particleCount; % actual fraction

    for i=1:particleCount

        %fill DATA array
         sim_coordinates(i*N-N+1:i*N, [1 3]) = i; %traj #
         sim_coordinates(i*N-N+1:i*N, 4) = 1:N; %frame #
         sim_coordinates(i*N-N+1:i*N, 5:6) = [particle(i).coordinates(:, 1),particle(i).coordinates(:, 2)];

    end





    case 'Switch'
        for i = 1:particleCount
            displacementsX = randn(1, N);
            displacementsY = randn(1, N);

            if numel(k)~=1
                states(:, i) = sim_simulateStates(N, k21, k12, tau);
                displacementsX(states(:, i)==0) = displacementsX(states(:, i)==0)*k(1);
                displacementsY(states(:, i)==0) = displacementsY(states(:, i)==0)*k(1);
                displacementsX(states(:, i)==1) = displacementsX(states(:, i)==1)*k(2);
                displacementsY(states(:, i)==1) = displacementsY(states(:, i)==1)*k(2);
            else
                displacementsX = displacementsX*k(1);
                displacementsY = displacementsY*k(1);
                states = zeros(N*i, 1);
            end

            %calcuate coordinates
            particle(i).coordinates(:, 1) = cumsum(displacementsX);
            particle(i).coordinates(:, 2) = cumsum(displacementsY);

            % move if random start
            if randomStart==1
                particle(i).coordinates(:, 1) = particle(i).coordinates(:, 1)+random('unif', -range/4, range/4);
                particle(i).coordinates(:, 2) = particle(i).coordinates(:, 2)+random('unif', -range/4, range/4);
            end

            %fill DATA array
            freetrack = find(sim_coordinates(:, 3)==0,1,'first');
            trackLength = length(particle(i).coordinates(:, 1));
            sim_coordinates(freetrack:freetrack+trackLength-1, 3) = i;
            sim_coordinates(freetrack:freetrack+trackLength-1, 1) = max(sim_coordinates(:, 1))+1;

            sim_coordinates(freetrack:freetrack+trackLength-1, 4) = 1:trackLength;
            sim_coordinates(freetrack:freetrack+trackLength-1, 5:6) = [particle(i).coordinates(:, 1),particle(i).coordinates(:, 2)];
            % waitbar(i/particleCount);                 
        end
   
        stats.actualF1 = length(find(states==0))/(N*i);
        
    case 'Domains'
                
        domainsResolution = length(domains(:, 1));
        k = k .* (domainsResolution/range);
        
        p12 = k12*tau; % probability to switch from 1 to 2 in time dt
        p21 = k21*tau; % probability to switch from 2 to 1 in time dt
        if strfind(simMode,'chaotic')
            numdomains = max(domains(:))+1;
            dommotion = repmat(rand(1, numdomains, 2)-0.5, [2, 1, 1]); 
            dommotion = rand(N, numdomains, 2)-0.5;
            % normalize motion to let every particle in every frame move by
            % the same magnitude, but with a different direction for every
            % domain
            dommotion = dommotion ./ repmat(sqrt(dommotion(:,:,1).^2 + dommotion(:,:,2).^2),[1,1,2]) .* k(1);
%             dommotion = ones(N,numdomains,2).*k(1);
            domrad = 2.*pi.*randn(N, numdomains,2);
%             domv = Motion.Vmean .*(ones(N, numdomains,2) + 1./4 .* randn(N, numdomains,2));
            domv = Motion.Vmean .*(ones(N, numdomains,2) + 1./4 .* ones(N, numdomains,2));
        end
        
        for i = 1:particleCount
            flag2 =0;
            while flag2 ~= 1
                new_x = round(random('unif', round(borderPercentage*domainsResolution), round((1-borderPercentage)*domainsResolution))); % don't seed on border
                new_y = round(random('unif', round(borderPercentage*domainsResolution), round((1-borderPercentage)*domainsResolution))); % don't seed on border
%                 new_x = (random('unif', round(borderPercentage*domainsResolution), round((1-borderPercentage)*domainsResolution))); % don't seed on border
%                 new_y = (random('unif', round(borderPercentage*domainsResolution), round((1-borderPercentage)*domainsResolution))); % don't seed on border

                particle(i).coordinates(1, 1) = new_x;
                particle(i).coordinates(1, 2) = new_y;
           
                if domains(round(new_y), round(new_x)) % watch out, inverted!
                    if D(2) ~= 0 && k12 ~= 0
                        flag2 = 1;
                        particle(i).state(1) = 2;
                    end
                else
                    if D(1) ~= 0 && k21 ~= 0
                        flag2 = 1;
                        particle(i).state(1) = 1;
                    end
                end
            end
            
            
            for j=2:N
                % waitbar((i*N-N+j)/(N*particleCount));  
                %check if coordinates are within domain or not
                x = particle(i).coordinates(j-1, 1);
                y = particle(i).coordinates(j-1, 2);

                if strfind(simMode,'chaotic')
                    if round(x) > Grid.sx
                        x = Grid.sx;
                    end
                    if round(y) > Grid.sx
                        y = Grid.sx;
                    end
                    if round(x) < 1
                        x = 1;
                    end
                    if round(y) < 1
                        y = 1;
                    end
                    mode = domains(round(x), round(y)) + 1;
                    fac = 1;%rand(1);
                    if strfind(Motion.MovMode,'brownian')
                        u = fac*dommotion(j,mode,1);
                        v = fac*dommotion(j,mode,2);
                        new_x = particle(i).coordinates(j-1, 1)+u;
                        new_y = particle(i).coordinates(j-1, 2)+v;
                        run = 1;
                        while new_x < 1  || new_x > range
                            if run > 10
                                u = 0;
                            else
                                u = fac/run*dommotion(j,mode,1);
                            end
                            new_x = particle(i).coordinates(j-1, 1)+u;
                            run = run + 1;
                        end
                        run = 1;
                        while new_y > range || new_y <1
                            if run > 10
                                v = 0;
                            else
                                v = fac/run*dommotion(j,mode,2);
                            end
                            new_y = particle(i).coordinates(j-1, 2)+v;
                            run = run + 1;
                        end
%                         plot(new_x,new_y,'x','color',colors(i,:)), hold on

                    elseif strfind(Motion.MovMode,'confined')
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 colors = jet(particleCount);
%                 figure(2)
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                        Fx = @(x,x0) - (1/(Motion.springconstant^2)) * (x - x0); % Diameter of potential in um
                        u = (D(1)*tau*Fx(particle(i).coordinates(j-1, 1)*Cam.pixelsize*1e6, particle(i).coordinates(1, 1)*Cam.pixelsize*1e6))  + ... % backstriking force
                            (fac*dommotion(j,mode,1)); % brownian
                        new_x = particle(i).coordinates(j-1, 1) + u;
                            
                        v = (D(1)*tau*Fx(particle(i).coordinates(j-1, 2)*Cam.pixelsize*1e6, particle(i).coordinates(1, 2)*Cam.pixelsize*1e6)) +...
                            (fac*dommotion(j,mode,2));
                        new_y = particle(i).coordinates(j-1, 2) + v;
                            
                        run = 1;
                        while new_x < 1  || new_x > range
                            if run > 10
                                u = 0;
                            else
                                u = u/10;
                            end
                            new_x = particle(i).coordinates(j-1, 1) + u;
                            run = run + 1;
                        end
                        run = 1;
                        while new_y > range || new_y < 1
                            if run > 10
                                v = 0;
                            else
                                v = v/10;
                            end
                            new_y = particle(i).coordinates(j-1, 2) + v;
                            run = run + 1;
                        end
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                         figure(2);
%                         plot(new_x,new_y,'x','color',colors(i,:)), hold on
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    elseif strfind(Motion.MovMode,'directed')
                        u = domv(1,mode,1).*tau.*cos(domrad(1,mode,1))+... % directed
                            (fac*dommotion(j,mode,1)); % brownian
                        new_x = particle(i).coordinates(j-1, 1) + u;
                            
                        v = domv(1,mode,2).*tau.*sin(domrad(1,mode,2))+...
                            (fac*dommotion(j,mode,2));
                        new_y = particle(i).coordinates(j-1, 2) + v;
                            
                        run = 1;
                        while new_x < 1  || new_x > range
                            if run > 10
                                u = 0;
                            else
                                u = u/10;
                            end
                            new_x = particle(i).coordinates(j-1, 1) + u;
                            run = run + 1;
                        end
                        run = 1;
                        while new_y > range || new_y <1
                            if run > 10
                                v = 0;
                            else
                                v = v/run;
                            end
                            new_y = particle(i).coordinates(j-1, 2) + v;
                            run = run + 1;
                        end
                    end
                    
%                     new_x = particle(i).coordinates(j, 1)+(1*dommotion(mode,1));
%                     new_y = particle(i).coordinates(j, 2)+(1*dommotion(mode,2));

                    particle(i).coordinates(j, 1) = new_x;
                    particle(i).coordinates(j, 2) = new_y;
                    particle(i).displacement(j, 1) = u;
                    particle(i).displacement(j, 2) = v;
                else
%                   if domains(round(y), round(x)) % if within domain, k2
%                       flag1 = 0;
%                       while flag1 ~= 1 % boundary condition
%                                 new_x = particle(i).coordinates(j, 1)+(randn(1)*k(1));
%                                 new_y = particle(i).coordinates(j, 2)+(randn(1)*k(1));
%                             if new_x<domainsResolution && new_y<domainsResolution && new_x>0 && new_y>0
%                                 
%                                 if domains(round(new_y), round(new_x)) ==0 % if new position is outside
%                                     if random('bino', 1, p21) && D(1)~=0; %...accept it with p21 probability
%                                         flag1 = 1;
%                                         particle(i).state(j+1) = 1; 
%                                     end
%                                 else % if new position is still inside
%                                     flag1 = 1;
%                                     particle(i).state(j+1) = 2; 
%                                 end
%                             end
%                       end
%                       particle(i).coordinates(j+1, 1) = new_x;
%                       particle(i).coordinates(j+1, 2) = new_y;
%                       
%                   
%                    else % if outside domain, k1
%                       flag1 = 0;
%                       while flag1 ~= 1 % boundary condition
%                                 new_x = particle(i).coordinates(j, 1)+(randn(1)*k(2));
%                                 new_y = particle(i).coordinates(j, 2)+(randn(1)*k(2));
%                             if new_x<domainsResolution && new_y<domainsResolution && new_x>0 && new_y>0
%                                 
%                                 if domains(round(new_y), round(new_x)) ==1 % if new position is inside
%                                     if random('bino', 1, p12) && D(2)~=0; %...accept it with p21 probability
%                                         flag1 = 1;
%                                         particle(i).state(j+1) = 2; 
%                                     end
%                                 else 
%                                     flag1 = 1;
%                                     particle(i).state(j+1) = 1; 
%                                 end
% 
%                             end
%                       end
%                       particle(i).coordinates(j+1, 1) = new_x;
%                       particle(i).coordinates(j+1, 2) = new_y;
%                       
%                   end
                end
            end    


            %fill DATA array
            freetrack = find(sim_coordinates(:, 3)==0,1,'first');
            trackLength = length(particle(i).coordinates(:, 1));
            sim_coordinates(freetrack:freetrack+trackLength-1, 3) = i;
            sim_coordinates(freetrack:freetrack+trackLength-1, 1) = max(sim_coordinates(:, 1))+1;
            sim_coordinates(freetrack:freetrack+trackLength-1, 4) = 1:trackLength;
            sim_coordinates(freetrack:freetrack+trackLength-1, 5:6) = ([particle(i).coordinates(:, 1),particle(i).coordinates(:, 2)]);
            sim_coordinates(freetrack:freetrack+trackLength-1, 7:8) = ([particle(i).displacement(:, 1),particle(i).displacement(:, 2)]);
            sim_state(freetrack:freetrack+trackLength-1, 1) = i;
            sim_state(freetrack:freetrack+trackLength-1, 2) = particle(i).state(:);
            

        end
%         particleCount
%         sum(sim_coordinates(:,5:6)~=0)
%         sum(sim_coordinates(:,5:6)==0)
        sim_coordinates(:, 5:6) = sim_coordinates(:, 5:6); %.*(range/domainsResolution);
        stats.actualF1 = 1;%sum(sim_state(:, 2)==1)/sum(sim_state(:, 2)~=0); % actual fraction
end
% close(h);

