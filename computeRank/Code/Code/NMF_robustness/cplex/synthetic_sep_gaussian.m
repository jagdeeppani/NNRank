function [A,A_orig,B,C,true_index,permute_vect] = synthetic_sep_gaussian(d,n,k,beta)
% for mizutani paper implementation input m=5000, n=250, k= 10, sigma
% between 0:0.1:0.5

alpha=unifrnd(0,1,k,1);
C=unifrnd(0,1,k,n);

M_dash=sample_dirichlet(alpha,d-k);
B=vertcat(eye(k),M_dash);
permute_vect=randperm(d);
B = B(permute_vect,:);

A_orig = B*C;

if (beta>0)
    N=normrnd(0,1,d,n);
    % Normalizing the Noise matrix
        cn_a = sqrt(sum(A_orig.^2));
        %cn_n = sqrt(sum(N.^2));
        N = N * (beta/sqrt(d)) * diag(cn_a);
A = A_orig + N ;

else
    A = A_orig;
end

true_index = find(permute_vect<=k);

end
