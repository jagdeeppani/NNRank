function  W = solvelp(A,M)
addpath /home/jagdeep/Desktop/NMF/Experiments/glpkmex-master;
% Finds W by minimizing the l1 residual error
% Min 1's
% s.t M*W(:,i)-A(:,i)<=s
%     M*W(:,i)-A(:,i)>=-s
%     W(:,i)         >=0


m = size(A,1);
n = size(A,2);
k = size(M,2);

disp('lp started');
 W = zeros(k,n);
% 
 f = vertcat(zeros(k,1),ones(m,1));
% 
A_ineq = horzcat(M,-1*eye(m));
 temp1 = -1*horzcat(M,eye(m));
 A_ineq = vertcat(A_ineq,temp1);
clear temp1;
fprintf('Number of variables of lp is %d\n',length(f));
lb = vertcat(zeros(k,1),-Inf*ones(m,1));
ub = Inf*ones(k+m,1);

ctype='U';
for i=1:size(A_ineq,1)-1
    ctype=[ctype 'U'];
end
vartype='C';
for i=1:length(f)-1
    vartype=[vartype 'C'];
end

fprintf('number of constraints are %d\n',size(A_ineq,1));

Time=zeros(n,1);
for i=1:n
   b_ineq= vertcat(A(:,i),-1*A(:,i));
   %size(b_ineq)
   %size(A_ineq)
   %temp1 = linprog(f,A_ineq,b_ineq,[],[],lb,ub);
   %[temp1, ~, ~, ~, ~] = cplexlp (f, A_ineq, b_ineq, [], [], lb, ub);
   lp_start=tic;
   %disp('glpk will start');
   [temp1,~, ~, ~] = glpk (f, A_ineq, b_ineq, lb, ub, ctype, vartype);%, sense, param)  
   Time(i)=toc(lp_start);
   %fprintf('solvelp: glpk took %f secs for %d variables with %d constraints\n',Time(i),length(f),size(A_ineq,1));
%    if mod(i,1)==0
%        fprintf('Time for %dth step is %d\n',i,Time(i));
%    end
   
   W(:,i) = temp1(1:k);
   %[obj,W(:,i),~] = lp_solve(f',A_ineq,b_ineq,-1*ones(1,length(b_ineq)));
end
fprintf('Avg time taken is %f\n',mean(Time));
fprintf('total time taken is %f\n',sum(Time)); 
disp('lp ends\n');
%fprintf('Objective is %f',obj);
end

%[obj, x]=lp_solve([-1, 2], [2, 1; -4, 4], [5, 5], [-1, -1], [], [], [1, 2]);
%[obj,x,~] = lp_solve(f',A_ineq,b_ineq,-1*ones(1,length(b_ineq)));
