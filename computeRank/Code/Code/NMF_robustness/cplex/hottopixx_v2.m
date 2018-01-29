function [K,X] = hottopixx_v2(M,epsilon,r,p,norma,verbose)
addpath /home/jagdeep/Desktop/NMF/Experiments/glpkmex-master;
% Hottopixx_cplex - A linear optimization model for near-separable NMF
%
% Solve with CPLEX the linear optimization model from 
% 
% V. Bittorf, B. Recht, and E. R� and  J.A. Tropp, Factoring 
% nonnegative matrices with linear programs, Advances in Neural Information 
% Processing Systems (NIPS '12), pp. 1223-1231, 2012. 
%
% 
% [K,X] = hottopixx_cplex(M,epsilon,r,p,norma,verbose)
%
% ****** Input ******
% M = WH + N : a (normalized) noisy separable matrix, that is, H = [I,H']P 
%              where I is the identity matrix, W >= 0 and H'>= 0 and their
%              columns sum to one, P is a permutation matrix, and
%              N is sufficiently small. 
% epsilon    : noise level ||N||_1
% r          : number of columns to be extracted. 
% p          : vector in the objective function, default: p = randn(n,1). 
% norma      : normalize=1 will scale the columns of M so that they sum to one,
%              hence matrix M will satisfy the assumption above for any
%              input nonnegative separable matrix M (default value). 
%              normalize=0, no scaling is performed. 
% verbose    : if true, display iteration information of CPLEX (default: false). 
%
% ****** Output ******
% K    : index set corresponding to the r largest diagonal entries of X. 
% X    : solution of the LP model (Hottopixx)

[m,n] = size(M);

if nargin < 6 || isempty(verbose)
    verbose = false;
end

if nargin < 5 || isempty(norma)
    norma = 1;
end

if nargin < 4 || isempty(p)
    p = randn(n,1);
end

if norma == 1
    % 1. Normalize the columns of M
    D = diag(1./(1e-16+sum(M))); M = M*D;
end

num_var_x = n*n;
num_var_sp = m*n;
num_var_sm = m*n;

num_vars = num_var_x + num_var_sp + num_var_sm;

% Aeq = [
%     % split M*X rowwise against +/- parts
%     kron(speye(n,n), M), speye(m*n, m*n),  -speye(m*n, m*n); ...
%     % Sum of diagonal values of X is r
%     vec(speye(n,n))', sparse(1, m*n), sparse(1,m*n);
%     ];

Aeq = [
    % split M*X rowwise against +/- parts
    kron(speye(n,n), M), speye(m*n, m*n),  -speye(m*n, m*n); ...
    % Sum of diagonal values of X is r
    vec(speye(n,n))', sparse(1, m*n), sparse(1,m*n);
    ];
beq = [
    reshape(M, m*n,1); 
    r;
    ];

Aineq = [
    % sum of +/- parts are bounded by epsilon
    sparse(n, n*n),  kron(speye(n,n), ones(1,m)), ...
    kron(speye(n,n), ones(1,m)) ...
    % i-th row in X is elementwise bounded by the diagonal element in that row
    eyeshiftmatrix(n), sparse(n*n -n,n*m), sparse(n*n-n,n*m)
    ];
bineq = [
    2 * epsilon * ones(n,1);
    zeros(n*n - n,1);
    ];


%Aineq = [Aineq;-1*Aineq];

f=[vec(diag(p));sparse(num_var_sp,1);sparse(num_var_sm,1)];
lb = zeros(num_vars,1);
ub = inf*ones(num_vars,1);
ub( vec(speye(n,n))==1 ) = 1;


A=vertcat(Aineq,Aeq);
b=vertcat(bineq,beq);

ctype='U';
for i=1:size(Aineq,1)-1
    ctype=[ctype 'U'];
end
for i=1:size(Aeq,1)
    ctype=[ctype 'S'];
end

vartype='C';
for i=1:length(f)-1
    vartype=[vartype 'C'];
end


%[solution, ~, ~, ~, ~] = cplexlp (f, Aineq, bineq, Aeq, beq, lb, ub); % x0, options)
tic;
[solution,~, ~, ~] = glpk (f, A, b, lb, ub, ctype, vartype);
fprintf('hottTopixx: glpk took %f secs for %d variables with %d constraints\n',toc,length(f),size(A,1));
%solution = linprog(f,Aineq,bineq,Aeq,beq,lb,ub);

% c = Cplex();
% c.Model.obj = [vec(diag(p));sparse(num_var_sp,1);sparse(num_var_sm,1)];
% c.Model.lb = zeros(num_vars,1);
% c.Model.ub = inf*ones(num_vars,1);
% c.Model.ub( vec(speye(n,n))==1 ) = 1;
% 
% c.Model.A = [Aeq;Aineq];
% c.Model.rhs = [beq; bineq];
% c.Model.lhs = [beq; -inf*ones(size(bineq))];
% c.Model.sense = 'minimize';
% 
% c.Param.lpmethod.Cur = 1;
% 
% if ~verbose
%     c.DisplayFunc = [];
% end
% 
% c.solve();
% 
% if verbose
%     fprintf('Objval: %.4e\n', c.Solution.objval);
%     fprintf('Solution time: %.1ds\n', c.Solution.time);
% end

%size(solution)
X = reshape(solution(1:n*n),n,n);
x = diag(X);

K = zeros(r,1);
for i = 1 : r
    [~,b] = max(x);
    K(i) = b; x(b) = -1;
end


%clear c;

end % of function hottopixx_cplex
