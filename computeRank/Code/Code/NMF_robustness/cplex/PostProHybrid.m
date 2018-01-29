function Kp = PostProHybrid(M,x,epsilon,r) 

% Hybrid post-processing procedure for LP-based algorithms for
% near-separale NMF 
% 
% *** Description ***
% Compute both post-processings from 
% (1) choose_large_diag.m, and 
% (2) PostProHottTopixx.m, 
%
% and keep the best solution according to min_{H >= 0} ||M-M(:,K)H||_F
% 
% See Algorithm 6 in N. Gillis and R. Luce, Robust Near-Separable 
% Nonnegative Matrix Factorization Using Linear Optimization, arXiv, 
% February 2013. 
%
% 
% Kp = PostProHybrid(M,x,epsilon,r) 
%
% ****** Input ******
% M            : m-by-n matrix (n data point in dimension m)
% x            : n-dimensional vector whose entries are in the interval [0,1]
% epsilon      : noise level 
% r            : number of columns to extract
%
% ****** Output ******
% Kp           : index set of the extracted columns. 

% First postprocessing: 
Kp1 = choose_large_diag(x, r); 
H1 = nnlsHALSupdt(M,M(:,Kp1)); 
error1 = norm(M-M(:,Kp1)*H1,'fro'); 

% Second postprocessing: 
Kp2 = PostProHottTopixx(M,x,epsilon,r,1); 
H2 = nnlsHALSupdt(M,M(:,Kp2)); 
error2 = norm(M-M(:,Kp2)*H2,'fro'); 

% Best one
if error1 <= error2
    Kp = Kp1;
else
    Kp = Kp2;
end 

end % of function PostProHybrid