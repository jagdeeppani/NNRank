% Post-processed successive projection algorithm (Post-SPA).  
% 
% *** Description ***
% At each step of the algorithm, the column of M maximizing ||.||_2 is 
% extracted, and M is updated by projecting its columns onto the orthogonal 
% complement of the extracted column (see FastSepNMF.m). 
% After r indices have been identified, the post-processing of Arora et al. 
% (A Practical Algorithm for Topic Modeling with Provable Guarantees, 
% ICML '13) is used to refine the solution. 
% 
% J = FastSepNMFpostpro(M,r,normalize) 
%
% ****** Input ******
% M = WH + N : a (normalized) noisy separable matrix, that is, W full rank, 
%              H = [I,H']P where I is the identity matrix, H'>= 0 and its 
%              columns sum to at most one, P is a permutation matrix, and
%              N is sufficiently small. 
% r          : number of columns to be extracted. 
% normalize  : normalize=1 will scale the columns of M so that they sum to one,
%              hence matrix H will satisfy the assumption above for any
%              nonnegative separable matrix M. 
%              normalize=0 is the default value for which no scaling is
%              performed. For example, in hyperspectral imaging, this 
%              assumption is already satisfied and normalization is not
%              necessary. 
%
% ****** Output ******
% J        : index set of the extracted columns. 

function J = FastSepNMFpostpro(M,r,normalize) 

[m,n] = size(M); 
if nargin <= 2, normalize = 0; end
if normalize == 1
    % Normalization of the columns of M so that they sum to one
    D = spdiags((sum(M).^(-1))', 0, n, n); 
    M = M*D; 
end

% Successive Projection Algorithm
J = FastSepNMF(M,r,normalize);  

% Post-Processing of Arora et al. 
for j = 1 : length(J)
    R = M; 
    for k = 1 : length(J)
        if k ~= j
            u = R(:,J(k))/norm(R(:,J(k))); 
            R = R - u*(u'*R); 
        end
    end
    [a, J(j)] = max( sum(R.^2) );
end