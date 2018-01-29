function pct = measureIndex(K, Ktilde)
% pct = measureIndex(K, Ktilde) -- Compute percentage of correctly
%   identified columns.
%
% Given a matrix K where each row contains the set of indices corresponding
% to a column of W (i.e., all indices of duplicates of the same column), 
% this function returns the percentage of such equivalence classes present 
% in the index vector Ktilde. 
%
% ****** Input ******
%   K       :  matrix of equivalence classes of columns of W.
%              K(k,:) holds all column indices corresponding to the k-th
%              column of W
%   Ktilde  :  vector of column indices to test, must not be larger than
%              the number of rows of K
% 
% ****** Output ******
%   pct     :  percentage of rows of K having an entry in Ktilde

% r is the number of rows of K, that is, columns of W
[r,s] = size(K);

% rtilde is the number of columns that we want to grade
rtilde = length(Ktilde);

if isempty(K)
    error('True index set K must not be empty');
end

if any(any(K <= 0))
    error('True index set K contains invalid indices');
end

if any(any(Ktilde <= 0))
    error('Test index set K contains invalid indices');
end

if length(unique(Ktilde)) < rtilde
    error('Test index set contains duplicates');
end

if length(unique(K)) < r*s
    error('True index set contains duplicates');
end

if rtilde > r
    error('Too many indices in Ktilde');
end

meas = 0;
for k = 1 : rtilde
    [a,~] = find(K==Ktilde(k)); 
    if ~isempty(a)
        meas = meas+1;
        K(a,:) = 0; 
    end
end

% Percentage of indices correctly extracted
pct = meas/r;

end % of function measureIndex