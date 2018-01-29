% compare linprog, hlpk and other methods

clear all;
close all;
m = 50;   % m is the vocab size
n = 25;    % n is the number of documents 
k = 5;      % number of topics

noise_level=0.3;
type = 2;        % 1=middlepoint (requires n >= r+r(r-1)/2), 2=Dirichlet.
scaling = 1.0;   % Different scaling of the columns of Mtilde.
normalize = 1.0; % Different normalization of the Noise.
density = 1.0;   % Proportion of nonzero entries in the noise matrix 
                     % (but at least one nonzero per column). 

                     
for iter_count = 1:10
[~, A, ~, ~, ~, anchor_indices_true] = synthetic_data_gillis_lp(n, m, k, ...
        noise_level, type, scaling, density, normalize);

A=A';        % A is a words x docs matrix
disp('Hottopixx glpk started');
tic    
anchor_indices = hottopixx(A', noise_level, k); 
Time1(1,iter_count)=toc;
disp('Hottopixx glpk done');


disp('Hottopixx linprog started');
tic    
anchor_indices = hottopixx_linprog(A', noise_level, k); 
Time1(2,iter_count)=toc;
disp('Hottopixx linprog done');

disp('LP-rho1 glpk started');
tic    
anchor_indices = LPsepNMF_oone(A', noise_level, k); 
Time2(1,iter_count)=toc;
disp('LP-rho1 glpk done');


disp('LP-rho1 linprog started');
tic    
anchor_indices = LPsepNMF_oone_linprog(A', noise_level, k); 
Time2(2,iter_count)=toc;
disp('LP-rho1 linprog done');



end

