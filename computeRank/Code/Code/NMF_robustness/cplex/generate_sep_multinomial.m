function [A,A_orig,B,C,permute_vect] = generate_sep_multinomial(d,n,k,c,etta1,etta2,m,~)
% A = M * W + N;
% Here m is the no of trials.
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



Dg = diag(1./sum(B)); B = B*Dg;

C = sample_dirichlet(etta2*ones(k,1),n); C=C';
%sum(B)
%size(B)
%sum(C)
%size(C)


permute_vect = randperm(d);
B = B(permute_vect,:);
A_orig = B * C;

%B
%C
%A_orig

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
%A
%N
fprintf('Synthetic data generated in %d secs\n',toc);
end