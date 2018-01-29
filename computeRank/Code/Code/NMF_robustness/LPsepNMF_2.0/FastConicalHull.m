function K = FastConicalHull(M,r)

% FastConicalHull - Fast Conical Hull Algorithm for Near-separable 
% Non-negative Matrix Factorization
% 
% *** Description ***
% It recursively extracts r columns of the input matrix M:  at each step, 
% it selects a column of M corresponding to an extreme ray of the cone 
% generated by the columns of M, and then projects all the columns of M 
% on the cone generated by the columns of M extracted so far. 
%
% This is our implementation of XRAY(max) from A. Kumar, V. Sindhwani, and 
% P. Kambadur, Fast Conical Hull Algorithms for Near-separable Non-negative 
% Matrix Factorization, International Conference on Machine Learning, 2013 
% (ICML '13) (see also arXiv:1210.1190). 
% 
% K = FastConicalHull(M,r) 
%
% ****** Input ******
% M = WH + N : a noisy separable matrix, that is, W >=0, H = [I,H']P where 
%              I is the identity matrix, H'>= 0, P is a permutation matrix, 
%              and N is sufficiently small. 
% r          : number of columns to be extracted. 
%
% ****** Output ******
% K        : index set of the extracted columns corresponding to extreme
%            rays of cone(M)

[m,n] = size(M); 
R = M; % residual 
p = ones(m,1); % as suggested in arXiv:1210.1190
K = []; 

for k = 1 : r
    % Extract an extreme ray
    normR = sum(R.^2); 
    [~,i] = max(normR); 
    [~,j] = max( (R(:,i)'*M)./(p'*M+1e-16) ); 
    % Remark: If 1e-16 is replaced with 1e-9, then it fails on the Swimmer
    % data set. 
    K = [K; j]; 
    
    % Update residual 
    if k == 1
        H = nnlsHALSupdt(M,M(:,K)); 
    else
        h = zeros(1,n); h(j) = 1; 
        H = [H; h]; 
        H = nnlsHALSupdt(M,M(:,K),H); 
    end
    R = M - M(:,K)*H; 
    % !Warning! R should not be computed explicitely in the sparse case.
    % We do it here for simplicity but this version is impractical for
    % large-scale sparse datasets (such as document datasets); but 
    % see arXiv:1210.1190. 
end

end % of function FastConicalHull