function [A,A_orig,M,W,permute_vect] = generate_synthetic(d,n,k,c,etta1,etta2,beta,normalize)
% A = M * W + N;
M=zeros(d,k);
tic
for i=1:k
    alpha=ones(d,1);
    start_ind = c*(i-1) + 1;
    end_ind = c*i;
    scaling_factor = ((d-c)*etta1)/(c*(1-etta1));
    alpha(start_ind:end_ind)=alpha(start_ind:end_ind) * scaling_factor; 
    M(:,i) = sample_dirichlet(alpha,1)';
    
end

W = sample_dirichlet(etta2*ones(k,1),n); W=W';
A_orig = M * W;
if (beta>0)
    % Normalizing the Noise matrix
    
    if normalize == 1
        N=normrnd(0,1,d,n);
        cn_a = sum(abs(A_orig));
        %cn_n = sum(abs(N));
        N = N * (beta/sqrt(d)) * diag(cn_a);
        
    elseif normalize == 2
        N=normrnd(0,1,d,n);
        cn_a = sqrt(sum(A_orig.^2));
        %cn_n = sqrt(sum(N.^2));
        N = N * (beta/sqrt(d)) * diag(cn_a);
    elseif normalize == 0
        %Individual element wise noise.
        Bil = max(M,[],2);
        sigma = 2*beta*sqrt(n)*repmat(Bil,1,n);
        N=normrnd(0,sigma,d,n);
        
    else
        error('Invalid normalization type');
    end
else
    N=zeros(d,n);
end

A = A_orig + N ;
permute_vect = randperm(d);
%A = A(permute_vect,:);
%A_orig = A_orig(permute_vect,:);
%M = M(permute_vect,:);
fprintf('Synthetic data generated in %d secs\n',toc);
fprintf('Rank of B is %d \n',rank(M));
end


