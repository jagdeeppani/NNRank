function K = choose_large_diag(diagX, nnrank)
% K = choose_large_diag(X, nnrank)
%
% Given the diagonal diagX of a solution X from a LP formulation, extract 
% 'nnrank' column indices based on the magnitudes of diagX.  
% The following rules are obeyed by the extraction:
%
% Extract the 'nnrank' indices with largest entries in diagX. But, 
% 
% If nnz(diagX) < r, all the corresponding nonzero entries are extracted 
% in K  (hence that |K| < nnrank). 
%
% If nnz(diagX == 1.0) > r, r indices corresponding to 1.0 values are
% picked at random.
% 
% ****** Input ******
%   diagX  : vector whose entries are in the interval [0,1]
%   nnrank : number of indices to extract
% 
% ****** Output ******
%   K      : index set extracted from diagX

[values, K] = sort(diagX, 'descend');
first_zero_idx = find(values==0, 1);
last_one_idx = find(values==1.0, 1, 'last');

if isempty(last_one_idx) || last_one_idx <= nnrank
    % Pick the at most r largest non-zero entries
    if isempty(first_zero_idx)
        K = K(1:nnrank);
    else
        K = K(1:min(nnrank, first_zero_idx-1));
    end
else
    % Pick r indices at random out of all 1.0-entries
    p = randperm(last_one_idx);
    K = K(p(1:nnrank));
end

end %of function choose_large_diag