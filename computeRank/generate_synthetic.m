function [A,A_orig,M,W,permute_vect] = generate_synthetic(d,n,k,c,etta0,etta2,beta,normalize)
% A = M * W + N;
M=zeros(d,k);
tic
for i=1:k
    alpha=ones(d,1);
    start_ind = c*(i-1) + 1;
    end_ind = c*i;
    scaling_factor = ((d-c)*etta0)/(c*(1-etta0));
    alpha(start_ind:end_ind)=alpha(start_ind:end_ind) * scaling_factor; 
    M(:,i) = sample_dirichlet(alpha,1)';
    
end

W = sample_dirichlet(etta2*ones(k,1),n); W=W';
A_orig = M * W;

%Adding noise 
if (beta>0)
    % Normalizing the Noise matrix
    
    if normalize == 1
        N=normrnd(0,1,d,n);
        cn_a = sum(abs(A_orig));
        %cn_n = sum(abs(N));
        N = N * (beta/sqrt(d)) * diag(cn_a);
        A = A_orig + N ;
        
    elseif normalize == 2
        N=normrnd(0,1,d,n);
        cn_a = sqrt(sum(A_orig.^2));
        %cn_n = sqrt(sum(N.^2));
        N = N * (beta/sqrt(d)) * diag(cn_a);
        A = A_orig + N ;
    elseif normalize == 0
        %Individual element wise noise.
        Bil = max(M,[],2);
        %sigma = 2*beta*sqrt(n)*repmat(Bil,1,n);
        sigma = beta*repmat(Bil,1,n);
        N=normrnd(0,sigma,d,n);
        A = A_orig + N ;
        
    else
        error('Invalid normalization type');
    end
else
    A = A_orig;
end

permute_vect = randperm(d);
A = A(permute_vect,:);
A_orig = A_orig(permute_vect,:);
M = M(permute_vect,:);

%Nsq = sqrt(sum(N.^2,1)) ./ sqrt(sum(A_orig.^2,1));
%fprintf('Average column wise error beta_tilde(ICML) is : %f\n', mean(Nsq));
%csvwrite('tsvdOutput/Nsq.csv',Nsq);

%eps3=0.05;
%pureRecords=sum(W>(1-eps3),2);
%csvwrite('tsvdOutput/pureRecords_mat.csv',pureRecords);
%eps1 = mean(pureRecords)/n;
%eps2_arr = subset_noise(N,Bil,eps1);
%fprintf('with eps3= %f, eps1 = %f and max eps2 =%f\n', eps3, eps1, max(eps2_arr));
%csvwrite('tsvdOutput/eps2_arr_mat.csv',eps2_arr);

%csvwrite('B_mat.csv',M);
%csvwrite('/home/local/ANT/pjagdeep/temp/C_mat.csv',W);
%csvwrite('/home/local/ANT/pjagdeep/temp/A_mat.csv',A);
%csvwrite('/home/local/ANT/pjagdeep/temp/A_orig_mat.csv',A_orig);

fprintf('Synthetic data generated in %d secs\n',toc);
fprintf('Rank of M is %d \n',rank(M));
end


