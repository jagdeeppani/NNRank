function [K, X] = LPsepNMF_linprog_routine(M, epsilon, r, p, model, verbose, varargin)

% LPsepNMF_cplex - A new linear optimization model for near-separable NMF
%
% Solve with CPLEX the following linear program  
% 
%  min_{X >= 0} p^T diag(X) 
%        such that ||M-MX||_1 <= rhs_rho*epsilon, (model = 'absolute')**
%                  ||M(:,i)||_1 X(i,j) <= ||M(:,j)||_1 X(i,i) for all i,j, 
%                  X(i,i) <= 1 for all i. 
%
% ( **For model = 'relative': 
%     ||M(:,j)-MX(:,j)||_1 <= rhs_rho*epsilon*||M(:,j)||_1 for all j. )
%
%
% See N. Gillis and R. Luce, Robust Near-Separable Nonnegative Matrix 
% Factorization Using Linear Optimization, arXiv, February 2013. 
%
% 
% [K,X] = LPsepNMF_cplex(M, epsilon, r, p, model, verbose, varargin)
%
% ****** Input ******
% M = WH + N : a noisy separable matrix, that is, W >= 0, H = [I,H']P 
%              where I is the identity matrix, H'>= 0, P is a permutation 
%              matrix, and N is sufficiently small. 
% epsilon    : noise level ||N||_1
% r          : number of columns to be extracted 
%              This is not a necessary input: it is otherwise recovered
%              according to Theorem 3 in paper above, that is, r=ceil(trace(X)). 
% p          : vector in the objective function, 
%              default: p = ones(n,1) + .01*(rand(n,1)-.5). 
% model      : 'absolute', the noise is assumed to be independent of the
%               norm of the column of M (see above)
%              'relative', the noise is assumed to be dependent of the
%               norm of the column of M as in Hottopixx (see above)
% verbose    : if true, display iteration information of CPLEX 
%              default: false. 
% varargin   : Other parameters: 
%              - ('rhs_rho', rho) with rho>0: value of parameter rho in the LP
%                default: rhs_rho = 1. 
%              - ('subset', K) with K in [1:n]: only allow non-zero diagonal 
%                entries of X in index set K, that is, enforce X(i,i) = 0 
%                for i not in K). Allows to solve the LP faster if we can
%                preselect some columns of M (e.g., using a fast algorithm). 
%                default: K = 1:n (that is, all columns).  
% 
% ****** Output ******
% K    : index set corresponding to the r largest diagonal entries of X. 
% X    : solution of the LP (see above)

[m,n] = size(M);

ip = inputParser;

ip.addParamValue('rhs_rho', 1, @(x) x >= 0);
ip.addParamValue('subset', 1:n, @(x) ~isempty(x));

if nargin < 6 || isempty(verbose)
    verbose = false;
end

if nargin < 5 || isempty(model)
    model = 'absolute';
end

if nargin < 4 || isempty(p)
     p = ones(n,1) + .01 * (rand(n,1) - .5);
end

ip.parse(varargin{:});
rhs_rho = ip.Results.rhs_rho;
subset = ip.Results.subset;

if ~strcmp(model, 'absolute') && ~strcmp(model, 'relative')
    error('The model parameter must be %s or %s', 'absolute', 'relative');
end

num_var_x = n*n;
num_var_sp = m*n;
num_var_sm = m*n;

num_vars = num_var_x + num_var_sp + num_var_sm;

% Precompute some norms
colnorms = zeros(n,1);
for j=1:n
    colnorms(j) = norm(M(:,j),1);
end

if strcmp(model, 'absolute')
    noise_weights = ones(n,1);
else
    noise_weights = colnorms;
end

Aeq = [
    % split M*X rowwise against +/- parts
    kron(speye(n,n), M), speye(m*n, m*n),  -speye(m*n, m*n); ...
    ];

beq = vec(M);

Aineq = [
    % sum of +/- parts are bounded by epsilon
    sparse(n, n*n),  kron(speye(n,n), ones(1,m)), ...
        kron(speye(n,n), ones(1,m)) ...
    % i-th row in X is elementwise bounded by the diagonal element in that
    % row
    eyeshiftmatrix(n, colnorms), sparse(n*n -n,n*m), sparse(n*n-n,n*m)
    ];
bineq = [
    rhs_rho * epsilon .* noise_weights;
    zeros(n*n - n,1);
    ];

f=[vec(diag(p));sparse(num_var_sp,1);sparse(num_var_sm,1)];
lb = zeros(num_vars,1);
ub = inf*ones(num_vars,1);

UBmat = zeros(n,n);
for k=1:length(subset)
    row_idx = subset(k);
    UBmat(row_idx, :) = inf;
    UBmat(row_idx, row_idx) = 1.0;
end
ub(1:n*n) = vec(UBmat);

solution = linprog(f,Aineq,bineq,Aeq,beq,lb,ub); 
X = reshape(solution(1:n*n),n,n);

% c = Cplex();
% c.Model.obj = [vec(diag(p));sparse(num_var_sp,1);sparse(num_var_sm,1)];
% c.Model.lb = zeros(num_vars,1);
% c.Model.ub = inf*ones(num_vars,1);
% 
% UBmat = zeros(n,n);
% for k=1:length(subset)
%     row_idx = subset(k);
%     UBmat(row_idx, :) = inf;
%     UBmat(row_idx, row_idx) = 1.0;
% end
% 
% 
% c.Model.ub(1:n*n) = vec(UBmat);
% 
% c.Model.A = [Aeq;Aineq];
% c.Model.rhs = [beq; bineq];
% c.Model.lhs = [beq; -inf*ones(size(bineq))];
% c.Model.sense = 'minimize';
% 
% if ~verbose
%     c.DisplayFunc = [];
% end
% 
% c.solve();
% 
% if verbose
%     fprintf('Solution status: %s\n', c.Solution.statusstring);
%     fprintf('Objval: %.4e\n', c.Solution.objval);
%     fprintf('Solution time: %.1ds\n', c.Solution.time);
% end
% 
%X = reshape(c.Solution.x(1:n*n),n,n); 

if nargin < 3 || isempty(r)
    % If r is not given as an input, use trace(X): this works well for
    % relatively low noise levels (otherwise less columns are extracted
    % -this makes sense since for high noise, less columns are needed to
    % reconstruct all columns of M up to that accuracy). 
    % (Also, this works badly when rho is large since the required 
    % accuracy = rho*epsilon.) 
	r = ceil( trace(X) ); 
end
K = choose_large_diag(diag(X), r);

%clear c;

end % of function LPsepNMF_cplex
