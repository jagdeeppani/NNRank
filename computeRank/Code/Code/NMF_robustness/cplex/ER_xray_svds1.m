% Ellipsoid Rounding implementation
% Input A,k,rho,    A is document x words matrix
function anchor_index=ER_xray_svds(A,noise_level,k)

%A=full(A); % As the input A sparse

rho=k;
%noise_level=0.1;
I=[];
[m,n]=size(A);
% disp('svd started');
% tic
% if issparse(A)
%     disp('matrix is sparse');
%     try
%         [~,D,V]= svds(A,k);
%     catch err
%         disp('svd failed');
%         anchor_index=0;
%         return
%     end
%         
%     
% else
%     disp('matrix is dense');
%         try
%         [~,D,V]= svd(A);
%     catch err
%         anchor_index=0;
%         return
%     end
% end
% disp('svd done');
% toc
% t=min(m,n);


increment_counter=2;
while (length(I)<k && increment_counter<50 && rho<n)
    disp('svd started');
    try
        [~,D,V]= svds(A,rho);
    catch err
        disp('svd failed');
        anchor_index=0;
        return
    end

    disp('svd done');
    
    P = D * V';
    %P = P(1:rho,:); % P is a matrix of size rxn
    disp('SOlveQ_Cutting_Plane started');
    L=solveQ_Cutting_Plane(P); 
    disp('SolveQ_Cutting plane done')
    
    if (L==0)
        disp('cycle detected in solveQ_Cutting_Plane');
        %anchor_index=0;
        rho=rho+increment_counter;
        increment_counter=increment_counter+1;
        continue;
    end
    
    count=1;
    I=[];
    for i=1:n
        temp = P(:,i)'*L*P(:,i);
        if temp>=0.98 && temp<=1.02
            I(count)=i;
            count=count+1;
        end
    end
        rho=rho+increment_counter;
        increment_counter=increment_counter+1;
        
end
%disp(I);
if increment_counter>=50
    anchor_index=0;
    return
end
if rho>n
    anchor_index=0;
    disp('rho valu exceeded the number of words');
    return
end


if length(I)>k
    anchor_index=I(fast_hull(A(:,I),noise_level,k));
else
    anchor_index=I;
end

end



