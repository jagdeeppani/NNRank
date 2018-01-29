function L=solveQ_Cutting_Plane(P,~,~,u_lim1,theta,etta)
[r,n] = size(P);
% fprintf('\nRank of P is %d\n',rank(P));
%S = randperm(n);
%S=S(1:ceil(n/10)); % Initialization outputs sorted index
disp('Initialization of S started');
S= ER_initialize_S(P);
disp('Initialization of S done');

S=sort(S);
% theta=0.99;
% etta=5;
done=0;
loop_counter=1;
tolerance=0.001;

while (~done && loop_counter < 15)
    
    P_temp=P(:,S);      % S contains the indices of columns of P which are in current S_k
    Pk=horzcat(P_temp,-P_temp);
    
    %size(Q)
    [L,~] = MinVolEllipse_v2(Pk, tolerance);    % This find the origin centered MVEE
    if L==0
        disp('MinVolEllipse didnt stop\n');
        return;
    end
    
    %disp('MVEE done');
    %flag1=zeros(n,1);
    
    delta=zeros(1,n);
    for i=1:n
        delta(i)= P(:,i)'*L*P(:,i);
    end
    
    if (sum(delta <= u_lim1))==n
        done=1;
    end
    
    
    %sum(delta>1)
    
    if (~done)
        count_F=1;
        F=[];
        for i=1:length(S)
            if delta(S(i))<=theta
                F(count_F)=S(i);
                count_F=count_F+1;
            end
        end
        S_comp=setdiff(linspace(1,n,n),S);  % returns sorted index
        [delta_S_comp,index1]=sort(delta(S_comp),'descend');
        S_comp=S_comp(index1);
        
        threshold_idx=find(delta_S_comp<=u_lim1,1);   % Instead of 1 we may take 1.1
        if threshold_idx>1
            index1=1:threshold_idx-1;
            S_comp=S_comp(index1);
        elseif threshold_idx==1
            index1=[];
            S_comp=S_comp(index1);
            
        else
            fprintf('All deltas are > 1\n');
            
        end
        
        no_of_points=ceil((n-2*r)/etta);
        length_S=length(S)
        %         G=[];
        if length(S_comp)>no_of_points
            G=S_comp(1:no_of_points);
        else
            G=S_comp;
        end
        S=union(setdiff(S,F),G);
    end
    
    
    clear delta;
    
    fprintf('loop counter inside SolveQcutting_plane is %d\n',loop_counter);
    loop_counter=loop_counter+1;
    
end
if (loop_counter>=100)
    L=0;
end

end

