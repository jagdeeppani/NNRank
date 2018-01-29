for i=1:max(fmatrix(:,1))+1
    for j=1:max(fmatrix(:,2)+1)
        A(fmatrix(),j)=fmatrix(i,j);
        
        
    end
end

for i=1: size(toy_17x100,1)
    toy_temp(toy_17x100(i,1)+1,toy_17x100(i,2)+1)=toy_17x100(i,3);
end

A=A';
count=1;
for i=1:size(A,1)
    for j=1:size(A,2)
        temp_mat_A(count,:)=[i-1,j-1,A(i,j)];
    end
end

    
        