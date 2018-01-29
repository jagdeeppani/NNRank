function [W,Z,temp1]= admm_nnlad_solver(A,M)
% This solves the L1 minimization problem 
% min ||A-M*W||_1  s.t  W>=0
% The method given here is from Extended XRAY Abhishek kumar etal.



stime=tic;
[m,n]=size(A);
k=size(M,2);

Z=zeros(m,n);
lambda = 0.001;
rho=1;
alpha=1;
I=eye(m);
max_iter=10;
iter=1;
tolerance=0.0001;

W_old=zeros(k,n);
while iter<max_iter 
    tic

    % Fix Z, solve for W
    
    W = nnlsHALSupdt(A-Z,M);
    temp1= norm(W_old-W,1);
    if  temp1< tolerance
        disp('W and W_old are close');
        break;
    end
    
    W_old=W;
    if mod(iter,1)==0
        fprintf('iteration: %d\n',iter);
    end

    % Fix W, solve for Z

    %Z_old=Z;
    L = A-M*W;
    %disp('ADMM started');
    for i=1:n
      
        Z(:,i) = lasso_admm_boyd_v2(I, L(:,i), lambda, rho, alpha);
        if mod(i,500)==0
        fprintf('subiteration %d done\n',i);
        end
    
    end
    
%disp('ADMM done');
    iter=iter+1;
    toc;
end
fprintf('Total time taken is %f\n',toc(stime));

end
