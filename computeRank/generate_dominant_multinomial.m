function [A,A_orig,B,C,permute_vect] = generate_dominant_multinomial(d,n,k,c,eta1,eta2,m,~)
% A = M * W + N;
% Here m is the no of trials.

B=zeros(d,k);
tic
for i=1:k
    alpha = ones(d,1);
    start_ind = c*(i-1) + 1;
    end_ind = c*i;
    scaling_factor = ((d-c)*eta1)/(c*(1-eta1));
    alpha(start_ind:end_ind) = alpha(start_ind:end_ind) * scaling_factor; 
    B(:,i) = sample_dirichlet(alpha,1)';    
end

Dg = diag(1./sum(B));
B = B*Dg;
permute_vect = randperm(d);
B = B(permute_vect,:);

C = sample_dirichlet(eta2*ones(k,1),n);
A_orig = B * C';

if (m>0)
    N = mnrnd(m,A_orig');
    N = (1/m)*N;
    N = N';
    N = N - A_orig;
    A = A_orig + N ;
else
    A = A_orig;
end

fprintf('Synthetic data generated in %d secs\n',toc);
%fprintf('Average number of unique words in a doc is %d \n',nnz(N)/n);
end