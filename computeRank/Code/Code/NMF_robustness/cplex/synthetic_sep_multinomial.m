function [A,A_orig,B,C,true_index,permute_vect] = synthetic_sep_multinomial(d,n,k,m)
% d words, n docs, k topics, B topic word matrix, C doc-topic matrix. m: no of samples for multinomial noise. lower m implies more noise.

alpha=unifrnd(0,1,k,1);
C=unifrnd(0,1,k,n);
Dg = diag(1./sum(C)); C = C*Dg;

M_dash=sample_dirichlet(alpha,d-k);
B=vertcat(eye(k),M_dash);
Dg = diag(1./sum(B)); B = B*Dg;
permute_vect=randperm(d);
B = B(permute_vect,:);

A_orig = B*C;

if (m>0)
    N = mnrnd(m,A_orig');
 %   N
    N = (1/m)*N;
    N = N';
  %  N
    N = N - A_orig;
    A = A_orig + N ;
else
    A = A_orig;
end

true_index = find(permute_vect<=k);

end
