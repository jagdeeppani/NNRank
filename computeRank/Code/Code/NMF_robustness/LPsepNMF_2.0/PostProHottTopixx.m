function Kp = PostProHottTopixx(Mi,x,epsilon,r,normalize) 

% Post-processing procedure, see Algorithm 4 in  
% Nicolas Gillis, Robustness Analysis of HottTopixx, a Linear Programming 
% Model for Factoring Nonnegative Matrices, arXiv:1211.6687. 
% 
% See also Algorithm 4 in N. Gillis and R. Luce, Robust Near-Separable 
% Nonnegative Matrix Factorization Using Linear Optimization, arXiv, February 2013. 
%
%
% *** Description ***
% Using the optimal solution X of an LP-based algorithm, extract vertices 
% by clustering the diagonal entries of X according to the distance 
% between the columns of Mi
%
% 
% Kp = PostProHottTopixx(M,x,epsilon) 
%
% ****** Input ******
% Mi           : m-by-n matrix (n data point in dimension m)
% x            : n-dimensional vector of weight
% epsilon      : noise level 
% r            : desired number of columns
% normalize    : normalize the columns of M, hence the clustering is made
%                by looking at the 'angles' (more formally, the distances 
%                between the normalized data points) between columns of M 
%               (this is important if the separable matrix is not normalized). 
% 
% ****** Output ******
% Kp           : index set of the extracted columns. 


Indices = find(x > 0); % Only keep indices with x > 0
M = Mi(:,Indices); x = x(Indices); 

if nargin >= 5 && normalize == 0
   % Normalize columns of M
   M = M*diag(1./sum(M));  
end


if nargin < 4
    r = ceil(sum(x)); 
else
    x = x/sum(x)*r;
end

[m,n] = size(M); 

if length(x) < r
    Kp = Indices;
    return;
end

if r > n
    error('The sum of the entries of x must be smaller than the number of data points.');
end

% Compute the distance matrix D(i,j) = ||M(:,i)-M(:,i)||_1
D = zeros(n,n); miniD = +Inf; 
for i = 1 : n
    D(i,1:i-1) = D(1:i-1,i)'; 
    D(i,i) = 0; 
    if i < n
        D(i,i+1:n) =  sum( abs( repmat(M(:,i),1,n-i) - M(:,i+1:n) ) ); 
        miniD = min(miniD, min(D(i,i+1:n))); 
    end
end

nu = max(2*epsilon,miniD) + 1e-6; 

K = find( x > r/(r+1) ); Kp = K; nup = nu ; 
nM = norm(M,1);

while length(K) < r && nu <= nM
    K = ClusterExtraction(D,x,nu); 
    if length(K) > length(Kp), Kp = K; nup = 0.5*nu; end
    nu = nu*2; 
end
 
if length(K) < r
    Kp = ClusterExtraction(D,x,nup);
end

Kp = Indices(Kp)'; 

end % of function PostProHottTopixx



function K = ClusterExtraction(D,x,nu) 

% *** Description ***
% Using the distance matrix D and the weights x, identify disjoint clusters 
% i in K s.t. S_i = {j | D(i,j) <= nu} with sum_{j in S_i} x(j) > r/(r+1). 
% If less than r indices are extracted, reduce r/(r+1). 
% 
% K = ClusterExtraction(D,x,nu)
%
% ****** Input ******
% D      : distance matrix, D(i,j) = ||M(:,i)-M(:,j)||_1
% x      : weight vector
% nu     : cluster diameter
% 
% ****** Output ******
% K      : index set extracted

[n,n] = size(D); 
S = double(D <= nu); 
w = S*x; 
K = []; r = round(sum(x)); 
wn = w; 
[mw,kw] = max(wn);  
maxiD = max(D(:)); 

while mw > r/(r+1)
    K = [K; kw]; 
    wn = wn - (S.*repmat(S(kw,:),n,1))*x; 
    [mw,kw] = max(wn); 
end
num_retries = 0;
K = []; wn = w; 
[mw,kw] = max(wn); 
while length(K) < r && num_retries < 10 
    K = [K; kw]; 
    wn = wn - ( S.*repmat(S(kw,:).*((maxiD-D(kw,:))/maxiD).^(0.1),n,1) )*x ; 
    wn(kw) = -1 ; 
    [mw,kw] = max(wn); 
    if num_retries < 9 && mw < 0 && length(K) < r
        num_retries = num_retries + 1;
        nu = nu/2;  
        K = []; 
        S = double(D <= nu); 
        w = S*x; wn = w;  
        [mw,kw] = max(wn); 
    end
end

end % of function ClusterExtraction
