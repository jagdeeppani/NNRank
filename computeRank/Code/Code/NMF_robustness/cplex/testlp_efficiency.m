clear all;
close all;
m = 100;
n = 50;
k = 20;

noise_level = 0.2;
type = 2;        % 1=middlepoint (requires n >= r+r(r-1)/2), 2=Dirichlet.
scaling = 1.0;   % Different scaling of the columns of Mtilde.
normalize = 1.0; % Different normalization of the Noise.
density = 0.25;   % Proportion of nonzero entries in the noise matrix 
                     % (but at least one nonzero per column). 
                

    [~, A, ~, ~, ~, K_true] = synthetic_data_gillis_lp(m, n, k, ...
        noise_level, type, scaling, density, normalize);
    
outpath = '/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch/tsvd';

[M,~] = TSVD(A,outpath,k);
disp('M found');
tic
%W = solvelp_cplex(A,M);
W = solvelp(A,M);

qp_time=toc;
fprintf('Finding W took %f secs',qp_time);
% disp('starting lp')
% tic
% W_lp = solvelp(A,M); 
% lp_time=toc;




% epsilon=0.2;
% tic
% [Ind_hot,X] = hottopixx_v2(fea,epsilon,k,[],1);
% W_hot = fea(:,Ind_hot);
% M_hot = solveqp(fea,W_hot);
% time_hot = toc;

A2 = A*spdiags(1./sum(A,1)',0,n,n);



Residual_l2_error = norm(A2-M*W,'fro')
Residual_l1_error = norm(A2-M*W,1)