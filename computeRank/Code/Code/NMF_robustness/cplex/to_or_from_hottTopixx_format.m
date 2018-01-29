function to_or_from_hottTopixx_format(ip_mat,flag)
% This code converts general matrices to hottTopixx format(tsv file) and
% vice versa



    
% From hottTopixx format to general format
load A_orig;

for i=1: size(A,1)
    A_temp(A(i,1)+1,A(i,2)+1)=A(i,3);
end



% From general to hottTopixx format
A_small=A';   % The rows of new A contains documents
count=1;
temp_mat_A_small=zeros(size(A_small,1)*size(A_small,2),3);
for i=1:size(A_small,1)
    for j=1:size(A_small,2)
        temp_mat_A_small(count,:)=[i-1,j-1,A_small(i,j)];
        count=count+1;
    end
end
dlmwrite('A_matrix_hottopixx_small2.tsv',temp_mat_A_small,'delimiter','\t','precision',4);




% code to get residual error and other measures after Hottopixx
K_test=[0 1 2 3 4 5 7 8 9 11 12 13 14 15 16 17 18 19 21 22 25 31 32 46 107];    % Feed the hottTopixx here
K_test=K_test+1;
W=A(K_test,:);
M = nnlsHALSupdt(A',W');
M=M';
residual_h=A-M*W;
residual_norm_orig_temp_h = norm(residual_h,1);
temp = 1- ( norm(residual_h,1) / norm(A,1) );
inverse_normalized_residual_norm=temp;

end