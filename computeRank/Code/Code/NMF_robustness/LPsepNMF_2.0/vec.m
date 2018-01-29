function v = vec(A)
% v = vec(A)  --  stack columns of A into a vector
%
% ****** Input ****** 
% A  :  an m-by-n matrix A
% 
% ****** Output ****** 
% v  :  an mn-dimensional vector with the columns of matrix A stacked
%       together

[m,n] = size(A);
v = reshape(A, m*n, 1);
end % of function vec
