% compare linprog, hlpk and other methods

clear all;
close all;
% m = 200;   % m is the vocab size
% n = 1000;    % n is the number of documents 
% k = 10;      % number of topics
% 
% noise_level=0.3;
% type = 2;        % 1=middlepoint (requires n >= r+r(r-1)/2), 2=Dirichlet.
% scaling = 1.0;   % Different scaling of the columns of Mtilde.
% normalize = 1.0; % Different normalization of the Noise.
% density = 1.0;   % Proportion of nonzero entries in the noise matrix 
%                      % (but at least one nonzero per column). 
% 
%                      
% [~, A, ~, ~, ~, anchor_indices_true] = synthetic_data_gillis_lp(n, m, k, ...
%         noise_level, type, scaling, density, normalize);
% 
% A=A';        % A is a words x docs matrix
    
load Reuters21578.mat;
clear trainIdx testIdx;
fea = fea(1:7285,:); % 7285 x 18933
gnd = gnd(1:7285);      % 7285 x 1
fea(:,find(sum(fea,1)<1))=[];

A=fea';
[m,n]= size(A)
k=10;
noise_level=0.3;
    
    
    
    

disp('spa started');    
anchor_indices = spa(A', noise_level, k);
W = A(anchor_indices,:);
% tic
% M_lp = solvelp(A',W'); % solving for 1 norm error and Rev(Inf,1) error are same. % solve lp or nnls, which is better
% Time(1)=toc;
% M_lp = M_lp';

tic
[M_admm,Z,temp1] = admm_nnlad_solver(A',W'); % solving for 1 norm error and Rev(Inf,1) error are same. % solve lp or nnls, which is better
Time(2)=toc;
M_admm = M_admm';


% res_norm_temp(1) = 1 - ( norm(A-M_lp*W,1) / norm(A,1) );   %To be updated
res_norm_temp(2) = 1 - ( norm(A-M_admm*W,1) / norm(A,1) );   %To be updated



% fprintf('%s : L1 residual norm : (%5.1f) \n', funs_str{alg_idx}, temp);
