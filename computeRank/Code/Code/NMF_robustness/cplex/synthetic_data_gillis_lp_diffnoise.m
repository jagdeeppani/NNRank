function [M, Mtilde, W, H, Noise, K] = ...
    synthetic_data_gillis_lp_diffnoise(m, n, r, epsilon, type, scaled, density, normalize, varargin)

% [M, Mtilde, W, H, Noise, K] = ...
%   synthetic_data(m, n, r, epsilon, type, scaled, density, normalize, ...)
%
% Generate an m-by-n nonnegative r-separable matrix M, and a noisy variant 
%
%   Mtilde = M + N = W * H + N
%
% where N is noise of level epsilon, meaning that
%
%   ||N||_1 <= epsilon .
%
% This function offers four different types of synthetic data sets, which
% can be set by the 'type' paramter:
%
%   type=1: "Middle Point" type: 
%           - The entries of W are drawn uniformly at random from the interval [0,1]
%           - H(:,1:r) = identity matrix to satisfy the separability assumption 
%           - H(:,r+1:N+r) = two nonzero entries per column (= 0.5), N=r(r-1)/2
%           - Remaining columns of H are generated following a Dirichlet distribution 
%             whose r parameters are chosen uniformly in [0,1]. 
% 
%   type=2: "Dirichlet" type: 
%           - H(:,1:r) = identity matrix to satisfy the separability assumption 
%           - Remaining columns of H are generated following a Dirichlet distribution 
%             whose r parameters are chosen uniformly in [0,1]. 
%
%   type=3: Ill-conditioned "Middle Point" type: 
%           Same as type 1 except that W is modified to be ill-conditioned
%           (condition number of 1e3).
% 
%   type=4: Ill-conditioned "Dirichlet" type: 
%           Same as type 2 except that W is modified to be ill-conditioned
%           (condition number of 1e3).
%
%   Eeac entry of the noise matrix N is generated following the normal 
%   distribution N(0,1), and then a proportion of about (1-density) entries 
%   are set to zero (see density below) and the columns of N are scaled to 
%   match the noise level epsilon (see normalize below)
%
% ****** Input ******
%   m        :  number of rows of M, Mtilde and W
%   n        :  number of columns of M, Mtilde and H
%   r        :  number of colums (resp. rows) of W (resp. H)
%   epsilon  :  noise level, see above (default: 0.0)
%   type     :  integer value in [1,4], sets the type of data, see above
%               (default: 1)
%   scaled   :  1 if columns of W, H are scaled, otherwise > 1 and the
%               columns of M are each multiplied by a number uniformly and
%               randomly between 1 and scaled (default: 1.0)
%   density  :  density of the noise, must be in [0,1], 1 means full.
%               It is guaranteed that each column of the Noise contains
%               at least one nonzero entry, even if spars==0.0.
%               See sprand's parameter 'density' for further information
%               (default: 1.0)
%   normalize: How to normalize the noise:
%               1: Some column columns of Noise has 1-norm == epsilon (def)
%               2: All column columns of Noise have 1-norm == epsilon
%               3: Some column columns of Noise has 2-norm == epsilon
%               4: All column columns of Noise have 2-norm == epsilon
%
% Further, the following options may be set in ('keyword', value) pairs:
%
%   'do_permute' :  true/false switch (default true)
%                   true: the separable columns appear at random positions
%                   false: the separable columns are the first r columns
%
% ****** Ourput ******
%   M      --  nonnegative matrix
%   Mtilde --  M + noise
%   W      --  W factor in M = W*H
%   H      --  H factor in M = W*H
%   K      --  each row of K contains a set of indices corresponding to the
%              same column of W. 

if nargin < 8 || isempty(normalize)
    normalize = 1;
end

if nargin < 7 || isempty(density)
    density = 1.0;
end

if nargin < 6 || isempty(scaled)
    scaled = 1.0;
end

if nargin < 5 || isempty(type)
    type = 1;
end

if nargin < 4 || isempty(epsilon)
    epsilon = 0.0;
end

ip = inputParser;
addParamValue(ip, 'do_permute', true, @(x) islogical(x));
ip.parse(varargin{:});

% Global switch that controls random permutation of the separable columns
% of M, N and H.  For debugging purpose, it may be of advantage to disable
% this permutation.
do_permute = ip.Results.do_permute;

if ~isreal(scaled) || scaled < 1.0
    error('Parameter scaled must be a real number no smaller than 1.0');
end

if ~any(type == [1,2,3,4])
    error('Paramter type must be an integer in [1,4]');
end

if ~any(normalize == [1,2,3,4])
    error('Paramter normalize must be an integer in [1,4]');
end

if ~isreal(epsilon) || epsilon < 0.0
    error('Parameter epsilon must be nonnegative');
end

% 1. Generation of W
% 1.a. W is randomly generated (hence typically well-conditioned)
if type == 1 || type == 2 
    W = rand(m,r);
% 1.b. W is ill-conditioned
elseif type == 3 || type == 4
    param = 3; % cond(W) is approximately 10^param
    beta = 10^(-param/(r-1)); 
    S = beta.^(0:r-1);
    W = rand(m,r); 
    [u,~,v] = svds(W,r);
    W = u*diag(S)*v'; 
end

% Scaling of the columns of W
D = diag(1./sum(W)); W = W*D; 

% 2. Generation of H and Noise 
% 2.a. H is such that there is a point in the middle of each pair of
%      columns of $W$. 
%      N is such that the columns of M located on the middle of two columns of W 
%      are moved in the direction oppositise to the mean of the columns of
%      W, the columns od W are not perturbed. 
if type == 1 || type == 3
    if n < r+r*(r-1)/2
        error('n has to be larger than r+r*(r-1)/2'); 
    end
    alpha = rand(r,1); 
    H = [eye(r) nchoose2(r)/2 sample_dirichlet(alpha,n-r-r*(r-1)/2)']; 
    K = (1:r)'; 
    M = W*H; 
    Noise = [zeros(m,r) M(:,r+1:end)-repmat(mean(W,2),1,n-r) ]; 
% 2.b. H is randomly generated (Dirichlet) but each basis element is repeated twice 
%      N is randomly generated (Guassian)
elseif type == 2 || type == 4
    if n < 2*r
        error('n has to be larger than 2r'); 
    end
    alpha = rand(r,1); 
    H = [eye(r) sample_dirichlet(alpha,n-r)']; 
    %H = [eye(r) eye(r) sample_dirichlet(alpha,n-2*r)']; ---with repetition
    K = [1:r]'; % with repetition: use K = [(1:r)' (r+1:2*r)']
    Noise = randn(m,n);
    
    %H(1:r,1:r)= diag(randperm(r)); % this replaces the identity by a diagonal matrix
    
    M = W*H;  
end

if scaled > 1 
        % Multiply each column of H and M by some random scalar between 1 and scaled
        sclamult = 1+(scaled-1)*rand(1,n); 
        H = H.*repmat(sclamult,r,1); 
        M = M.*repmat(sclamult,m,1); 
        if type == 1 || type == 3 
            % We want to preserve the type of perturbation: 
            % columns of M are move toward the outside of the simplex.
            Noise = Noise.*repmat(sclamult,m,1); 
        end
end

% Make the noise sparse according to the density paramteter.
% It is guaranteed that at least one entry in every nonzero column of the
% Noise matrix remains a structural nonzero.
if density < 1.0
    mask = spones( ...
        sprand(m,n,density) + sparse(randi(m, n, 1), 1:n, ones(n,1), m, n));
    Noise = Noise .* mask;    
else
    % We return the full Noise matrix: leave it untouched
end

% Normalization according to:
% 1: ||Noise(:,j)||_1 == epsilon*||M(:,j)||_1  for at least one j
% 2: ||Noise(:,j)||_1 == epsilon ||M(:,j)||_1  for all j
% 3: ||Noise(:,j)||_2 == epsilon*||M(:,j)||_2  for at least one j
% 4: ||Noise(:,j)||_2 == epsilon ||M(:,j)||_2  for all j
%
% Note that in the type 1/3 case, the noise must stay zero in non-separable
% columns.

if normalize == 1

   if (beta>0)
    N=normrnd(0,1,n,m);
    % Normalizing the Noise matrix
    
        cn_a = sqrt(sum(A_orig.^2));
        %cn_n = sqrt(sum(N.^2));
        N = N * (beta/sqrt(d)) * diag(cn_a);
    else
        error('Invalid normalization type');
    end
else
    N=zeros(d,n);
end


elseif normalize == 2
    cn_m = sum(abs(M));
    cn_n = sum(abs(Noise));
    if type==1 || type==3
        % Then the first r columns of Noise are zero.  We do not want to
        % mess with 0 * inf computations below, setting the norms to 1.0
        % is safe, for that the norms remain zero anyway.
        cn_n(K) = 1.0;
    end
    Noise = Noise * epsilon * diag(cn_m./cn_n);
elseif normalize == 3
    nM = max(sqrt(sum(M.^2)));
    nN = max(sqrt(sum(Noise.^2))); 
    Noise = epsilon*Noise/nN*nM; 
elseif normalize == 4
    cn_m = sqrt(sum(M.^2));
    cn_n = sqrt(sum(Noise.^2));
    if type==1 || type==3
        % Then the first r columns of Noise are zero.  We do not want to
        % mess with 0 * inf computations below, setting the norms to 1.0
        % is safe, for that the norms remain zero anyway.
        cn_n(K) = 1.0;
    end
    Noise = Noise * epsilon * diag(cn_m./cn_n);
else
    error('Probably false input argument checks, fixme');
end

Mtilde = M+Noise;

if do_permute
    p = randperm(n);
    q(p) = 1:n;
    M = M(:,p);
    Noise = Noise(:,p);
    Mtilde = Mtilde(:,p);
    H = H(:,p);
    [~, ncol] = size(K);
    for k=1:ncol
        K(:,k) = q(K(:,k));
    end
end

end % of function synthetic_data