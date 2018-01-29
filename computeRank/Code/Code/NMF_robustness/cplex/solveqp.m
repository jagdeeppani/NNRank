function  W = solveqp(A,M)

n=size(A,2);
r=size(M,2);
W = zeros(r,n);
opts=optimoptions('quadprog','Display','off');
disp('Quadprog started');
for i=1:n
     W(:,i) = quadprog(M'*M,-M'*A(:,i),[],[],[],[],zeros(r,1),Inf*ones(r,1),[],opts); 
end
disp('Quadprog ended');

%quadprog(H,f,A,b,[],[],lb,[],[],opts);