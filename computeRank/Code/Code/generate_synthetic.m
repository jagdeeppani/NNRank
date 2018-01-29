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
% disp('M found');
% tic
% val=zeros(1,k);
% for i=1:k
%     start_ind = c*(i-1) + 1;
%     end_ind = c*i;
%     val(i)=sum(M(start_ind:end_ind,i));
% end

% for i=1:k
%     start_ind = c*(i-1) + 1;
%     end_ind = c*i;
%     temp = M(start_ind:end_ind,i);
%     M(start_ind:end_ind,i) = temp*etta1/sum(temp);
%     
%     rest_ind = setdiff(1:d,start_ind:end_ind);
%     temp2 = M(rest_ind,i);
%     M(rest_ind,i) = temp2*(1-etta1)/sum(temp2);
%     
% end

% val1=zeros(1,k);
% for i=1:k
%     start_ind = c*(i-1) + 1;
%     end_ind = c*i;
%     val1(i)=sum(M(start_ind:end_ind,i));
% end
% toc

W = sample_dirichlet(etta2*ones(k,1),n); W=W';
A_orig = M * W;
if (beta>0)
    N=normrnd(0,1,d,n);
    % Normalizing the Noise matrix
    
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
permute_vect = randperm(d);
A = A(permute_vect,:);
A_orig = A_orig(permute_vect,:);
M = M(permute_vect,:);
fprintf('Synthetic data generated in %d secs\n',toc);
end


