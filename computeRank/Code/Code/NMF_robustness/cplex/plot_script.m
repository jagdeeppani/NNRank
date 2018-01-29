
    
% Synthetic Catchwords 
    load('res_norm_orig_s_normalized_final.mat');
    funs_str = {'Heur-SPA','SPA','XRAY','UTSVD'};
    x = 0:0.05:0.4;
    y = res_norm_orig_s_normalized_final';
    plot(x,y(:,1),'g-+',x,y(:,2),'b-o',x,y(:,3),'c-s',x,y(:,4),'m-<');%,x,res_norm_L1_final(:,5), ...
        %'m-.x',x,final_residual_norm(:,6),'y-.>',x,final_residual_norm(:,7),'b-.<',x,final_residual_norm(:,8),'m-.+');
    grid on;
    hold on;
    legend(funs_str{1},funs_str{2},funs_str{3},funs_str{4});%,funs_str{5},funs_str{6},funs_str{7},'un-normalized tsvd');
    title('Ls Residual for $A$ ( c = 10, $\alpha$ = 0.01) ');
    xlabel('Noise level ($\beta$)');
    ylabel('Ls Residual norm');
    
    
    x = 0:0.05:0.4;
    title1='$\ell_{s}$ Residual for $MW$';
    %load('res_norm_orig_s_normalized_final.mat');
    y = res_norm_orig_s_normalized_final';
    plot_full_synthetic_catchwords(x,y);
    
    
    
    
    % Synthetic Separability
    
    funs_str = {'PrecSPA','PostPrecSPA','HottTopixx','SPA', 'XRAY','LP-rho1','true-K','UTSVD'};
    x = linspace(0.01,1,40);
    load('res_norm_s_A_final.mat');
    y = res_norm_s_A_final';
    plot(x,y(:,1),'g-+',x,y(:,2),'b-o',x,y(:,3),'c-s',x,y(:,4),'r-o',x,y(:,5),'g--',x,y(:,6),'k-+',x,y(:,7),'r-d',x,y(:,8),'m-<');
    grid on;
    hold on;
    legend(funs_str{1},funs_str{2},funs_str{3},funs_str{4},funs_str{5},funs_str{6},funs_str{7},funs_str{8});
    title('$\ell_{s}$ Residual for $A$ (Middle point, sparse) ');
    xlabel('Noise level ($\epsilon$)');
    ylabel('$\ell_{s}$ Residual norm');
    
    
    x = linspace(0.01,1,40);
    title1='$\ell_{s}$ Residual for $A$ (Middle point, sparse)';
%     load('res_norm_s_Aorig_final.mat');
    y = res_norm_s_A_final';
    plot_full_synthetic_separable_large(x,y,title1);
    
    
    
    
    
        funs_str = {'PrecSPA','PostPrecSPA','HottTopixx','SPA', 'XRAY','LP-rho1','true-K','UTSVD'};
    x = linspace(0.01,1,40);
    load('time_final.mat');
    y = time_final';
    plot(x,y(:,1),'g-+',x,y(:,2),'c-s',x,y(:,4),'r-o',x,y(:,5),'g--',x,y(:,7),'r-d',x,y(:,8),'m-<');    
grid on;
    hold on;
    legend(funs_str{1},funs_str{2},funs_str{4},funs_str{5},funs_str{7},funs_str{8});
    title('$\ell_{2}$ Residual for $A$ (Dirichlet, sparse) ');
    xlabel('Noise level ($\epsilon$)');
    ylabel('$\ell_{2}$ Residual norm');
    
    
    
    % Semi-Synthetic
   funs_str = {'Heur-SPA','SPA','XRAY','UTSVD'};
   x=linspace(0,0.5,11);
%     load('res_norm_L2_Aorig_final_complete.mat');
    y = res_norm_s_Aorig_normalized_final';
    plot(x,y(:,1),'g-+',x,y(:,2),'r-o',x,y(:,3),x,y(:,4),'m-<');
    grid on;
    hold on;
    legend(funs_str{1},funs_str{2},funs_str{3},funs_str{4});
    title('$\ell_{s}$ Residual for $A$ (20,000 documents) ');
    xlabel('Noise level ($\epsilon$)');
    ylabel('$\ell_{s}$ Residual norm');
    
    
    
    
    
    %Clustering Plot
    funs_str = {'Heur-SPA','SPA','XRAY','UTSVD'};
    x = 0:1:10
%     load nmi_final.mat
    y=nmi_val';
    plot(x,y(:,1),'g-+',x,y(:,2),'r-o',x,y(:,3),'b-s',x,y(:,4),'m-<');
    grid on;
    hold on;
    legend(funs_str{1},funs_str{2},funs_str{3},funs_str{4});
    title('Clustering performance for 20 Newsgroups');
    xlabel('Iterations');
    ylabel('Normalized Mutual Information (NMI)');
    
        
    title1='Clustering performance for 20 Newsgroups';
%     load('res_norm_orig_s_normalized_final.mat');
x = 0:2:20;    
y = nmi_val';
z = y(:,end);
y(:,end)=[];
x1(2:11) = 1:2:19;
x1(1)=0;
plot_clustering_v2(x,y);
hold on;
plot(x1,z);
% plot_clustering_utsvd(x1,z);

x = 0:2:20; 
% x=Total_timef';
y = nmi_valf';
plot_clustering_v2_random(x,y);
% hold on;
% plot(x1,z);



    
    
    
    
    % Classification Plot
    funs_str = {'SPA','XRAY','UTSVD','Full-features','Heur-SPA'};
    x=50:50:350
    y=acc4;
    plot(x,y(1,:),'c-s',x,y(2,:),'r-o',x,y(3,:),'m-<',x,y(4,:),'b-+');
    grid on;
    hold on;
    legend(funs_str{1},funs_str{2},funs_str{3},funs_str{4},funs_str{5});
    title('SVM accuracy with selected features (6 iterations)');
    xlabel('No of features extracted');
    ylabel('SVM Accuracy');
    
    
    funs_str = {'UTSVD','SPA','XRAY','Heur_SPA','random'};
    x=Total_time;
    y=svm_accuracy;
    plot(x,y(1,:),'c-s',x,y(2,:),'r-o',x,y(3,:),'m-<',x,y(4,:),'b-+');
    grid on;
    hold on;
    legend(funs_str{1},funs_str{2},funs_str{3},funs_str{4},funs_str{5});
    title('SVM accuracy with selected features (6 iterations)');
    xlabel('No of features extracted');
    ylabel('SVM Accuracy');
    
   x = Total_time';
   x=0:2:20
   y = svm_accuracy';
   plot_classification_v2(x,y);

    
    
    
    
    
    
    
    
     
    
    
    
    
    
    
    
    
    
    
    
    
    
    