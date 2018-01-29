% This code 

A=A';
count=1;
temp_mat_A=zeros(size(A,1)*size(A,2),3);
for i=1:size(A,1)
    for j=1:size(A,2)
        temp_mat_A(count,:)=[i-1,j-1,A(i,j)];
        count=count+1;
    end
end

