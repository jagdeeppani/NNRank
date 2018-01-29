function minibench

clc; 

% The problem size of the synthetic data set
m = 20;
n = 55;
r = 10;

% Parameters for generating the noisy separable matrix Mtilde
% ---> see synthetic_data.m for more details 
noise_level = 0.2;
type = 2;        % 1=middlepoint (requires n >= r+r(r-1)/2), 2=Dirichlet.
scaling = 1.0;   % Different scaling of the columns of Mtilde.
normalize = 1.0; % Different normalization of the Noise.
density = 0.1;   % Proportion of nonzero entries in the noise matrix 
                 % (but at least one nonzero per column). 

[~, Mtilde, ~, ~, ~, K_true] = synthetic_data(m, n, r, ...
    noise_level, type, scaling, density, normalize);


disp('************************************************************************************'); 
disp('          Comparing near-separable NMF algorithms on a synthetic data set'); 
disp('************************************************************************************'); 
disp(' Properties of the generated noisy separable matrix: '); 
if type == 1
    disp('   - Type                : Middle Points'); 
elseif type == 2
    disp('   - Type                : Dirichlet'); 
elseif type == 3
    disp('   - Type                : Ill-conditioned Middle Points');     
elseif type == 4
    disp('   - Type                : Ill-conditioned Dirichlet'); 
end
fprintf('   - Dimensions          : %d by %d\n',m,n); 
fprintf('   - Factorization rank  : %d\n',r); 
fprintf('   - Noise level         : %d%%\n',100*noise_level); 
fprintf('   - Density of the noise: %d%%\n',100*density); 
disp('************************************************************************************'); 

% The near-separable NMF algorithms we want to test
funs = {
    @hottopixx, ...
    @spa, ...
    @fast_hull,...
    @LPsepNMF_oone, ...
    @LPsepNMF_otwo, ...
    };

% Descriptive names for the algorithms
funs_str = {
    'Hottopixx', ...
    'SPA',...
    'XRAY',...
    'LP-rho1', ...
    'LP-rho2', ...
    };

fprintf('%19s: ', 'true index set');
disp(sort(K_true)');

num_algs = length(funs);
for alg_idx = 1:num_algs
    fun = funs{alg_idx};
    K_test = fun(Mtilde, noise_level, r); 
    K_test = reshape(K_test, 1, length(K_test));  
    pct = 100 * measureIndex(K_true, K_test);
    fprintf('%10s (%5.1f%%): ', funs_str{alg_idx}, pct);
    disp(sort(K_test));
end

disp('************************************************************************************'); 

end % of minibench


% This function is a wrapper around the original LP model (Hottopixx) by 
% Bittorf, Recht, Ré & Tropp
function [K, X] = hottopixx(M, epsilon, r)
[~, X] = hottopixx_cplex(M, epsilon, r, [], 1, false);
K = PostProHybrid(M,diag(X),epsilon,r); 
end

% Wrapper for the fast conical hull algorithm by Kumar, Sindhwani & Kambadur
function [K,X] = fast_hull(M, ~, r)
K = FastConicalHull(M,r);
X = [];
end

% Wrapper for fast recursive algorithm for separable NMF by Gillis & Vavasis
function [K,X] = spa(M, ~, r)
K = FastSepNMF(M,r,1);
X = [];
end

% Wrapper for our new LP model with rho=1
function [K, X] = LPsepNMF_oone(M, epsilon, r)
[~, X] = LPsepNMF_cplex(M, epsilon, r, [], 'absolute', false,...
    'rhs_rho', 1.0);
K = PostProHybrid(M,diag(X),epsilon,r); 
end

% Wrapper for our new LP model with rho=2
function [K, X] = LPsepNMF_otwo(M, epsilon, r)
[~, X] = LPsepNMF_cplex(M, epsilon, r, [], 'absolute', false,...
    'rhs_rho', 2.0);
K = PostProHybrid(M,diag(X),epsilon,r); 
end