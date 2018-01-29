function [A_orig,A,W,M,true_index] = synthetic_data_mizutani(m,n,k,sigma)
% for mizutani paper implementation input m=5000, n=250, k= 10, sigma
% between 0:0.1:0.5

alpha=unifrnd(0,1,k,1);
M_dash=sample_dirichlet(alpha,m-k);
W=unifrnd(0,1,k,n);
N=normrnd(0,sigma,m,n);
M=vertcat(eye(k),M_dash);
A_orig = M*W;
A = A_orig + N ;
pi1=randperm(m);
A = A(pi1,:);
true_index=find(pi1<=k);

end
