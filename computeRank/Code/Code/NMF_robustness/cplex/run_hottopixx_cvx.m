clear all;
close all;

m = 50;
n = 100;
k = 10;

noise_level=0.5;
type = 2;        % 1=middlepoint (requires n >= r+r(r-1)/2), 2=Dirichlet.
    scaling = 1.0;   % Different scaling of the columns of Mtilde.
    normalize = 1.0; % Different normalization of the Noise.
    density = 0.9;   % Proportion of nonzero entries in the noise matrix 
                     % (but at least one nonzero per column). 
                     
                     
[~, A, ~, ~, ~, K_true] = synthetic_data_gillis_lp(m, n, k,noise_level, type, scaling, density, normalize);      

disp('hottopixx started');
tic;
[K_test,X] = HottTopixx_cvx(A,noise_level,k);
disp('hottopixx done')
toc
M=A(:,K_test);
tic
W = nnlsHALSupdt(A,M);
toc

