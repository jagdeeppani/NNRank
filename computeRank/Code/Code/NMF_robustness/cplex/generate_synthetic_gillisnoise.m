function [A,A_orig,B,C,permute_vect] = generate_synthetic_gillisnoise(d,n,k,c,etta1,etta2,beta,normalize)
% A = M * W + N;

B=zeros(d,k);
tic
for i=1:k
    alpha=ones(d,1);
    start_ind = c*(i-1) + 1;
    end_ind = c*i;
    scaling_factor = ((d-c)*etta1)/(c*(1-etta1));
    alpha(start_ind:end_ind)=alpha(start_ind:end_ind) * scaling_factor; 
    B(:,i) = sample_dirichlet(alpha,1)';
    
end

C = sample_dirichlet(etta2*ones(k,1),n); C=C';
permute_vect = randperm(d);
B = B(permute_vect,:);

A_orig = B * C;
if (beta>0)
    N=normrnd(0,1,d,n);
    % Normalizing the Noise matrix
    nM = max(sum(abs(M))); 
    nN = max(sum(abs(Noise))); 
    Noise = epsilon*Noise/nN*nM; 
    
    
    if normalize == 1
        cn_a = sum(abs(A_orig));
        %cn_n = sum(abs(N));
        N = N * (beta/sqrt(d)) * diag(cn_a);
        
    elseif normalize == 2
        cn_a = sqrt(sum(A_orig.^2));
        %cn_n = sqrt(sum(N.^2));
        N = N * (beta/sqrt(d)) * diag(cn_a);
    else
        error('Invalid normalization type');
    end
else
    N=zeros(d,n);
end

A = A_orig + N ;
fprintf('Synthetic data generated in %d secs\n',toc);
end


