function DomainSim = domains_chaotic(Grid, DomainSim)

% even though we need a bigger image in the end, we make it in small scale
% for the sake or performance. Faterwards, we can scale the resulting the
% image up by means of MATLAB's imresize(). The resulting resolution is
% good enough for our purpose.


num = DomainSim.numberOfdomains;
rad = DomainSim.diameter/2;

% initialize
DomainSim.domains   = zeros(Grid.sx, Grid.sy, Grid.sz);
DomainSim.Centroids = zeros(num, 3);
DomainSim.Radius    = zeros(num, 1);

[x, y, z]= meshgrid(1:Grid.sx, 1:Grid.sy, 1:Grid.sz);
vec      = 1:num;
vec      = vec(randperm(length(vec)));

for k = 1:num
    DomainSim.Radius(k)       = rad + randn * DomainSim.variation / 2;
    
    DomainSim.Centroids(k, 1) = ceil(random('unif', 0, Grid.sx, 1));
    DomainSim.Centroids(k, 2) = ceil(random('unif', 0, Grid.sy, 1));
    DomainSim.Centroids(k, 3) = ceil(random('unif', 0, Grid.sz, 1));
    
    d = ((x-DomainSim.Centroids(k, 1)).^2 + ...
        (y-DomainSim.Centroids(k, 2)).^2 + ...
        (z-DomainSim.Centroids(k, 3)).^2) <= DomainSim.Radius(k).^2;
        
    DomainSim.domains(d) = vec(k);
end

% scale up if necessary
if Grid.sampling ~= 1
    DomainSim.domains = imresize3(DomainSim.domains, Grid.sampling, 'nearest');
end

% SliderDemo(mat3Dcell(domains))

