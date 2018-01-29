% Test on the Swimmer Data Set (cf. Section 5.4 of `Robust Near-Separable 
% Nonnegative Matrix Factorization Using Linear Optimization', Gillis and 
% Luce.) 

close all; clear all; clc; 
load('SwimmerDatabase.mat'); 
[m,n] = size(M); 
r = 16; epsilon = 0.1;  

% ******** Hottopixx ********
fprintf('Hottopixx started...'); e = cputime; 
[~, X] = hottopixx_cplex(M, epsilon, r, [], 1, false);
Khot = PostProHybrid(M,diag(X),epsilon,r); 
timhot = cputime - e; 
Hhot = nnlsHALSupdt(M,M(:,Khot)); 
errorhot = norm(M-M(:,Khot)*Hhot,'fro');  
fprintf('Done in %2.2f seconds with error %2.2f. \n', timhot, errorhot); 

% ******** SPA ********
fprintf('SPA started...'); e = cputime; 
Kspa = FastSepNMF(M,r,1); 
timspa = cputime - e; 
Hspa = nnlsHALSupdt(M,M(:,Kspa)); 
errorspa = norm(M-M(:,Kspa)*Hspa,'fro');  
fprintf('Done in %2.2f seconds with error %2.2f. \n', timspa, errorspa); 

% ******** XRAY ********
fprintf('XRAY started...'); e = cputime; 
Kx = FastConicalHull(M,r)'; 
timx =  cputime - e; 
Hx = nnlsHALSupdt(M,M(:,Kx)); 
errorx = norm(M-M(:,Kx)*Hx,'fro');  
fprintf('Done in %2.2f seconds with error %2.2f. \n', timx, errorx); 

% ******** LP rho = 1 ********
fprintf('LP rho = 1 started...'); e = cputime; 
[~, X] = LPsepNMF_cplex(M, epsilon, r, [], 'relative', false,...
    'rhs_rho', 1.0);
timlp = cputime - e; 
Klp = PostProHybrid(M,diag(X),epsilon,r); 
Hlp = nnlsHALSupdt(M,M(:,Klp)); 
errorlp = norm(M-M(:,Klp)*Hlp,'fro'); 
fprintf('Done in %2.2f seconds with error %2.2f. \n', timlp, errorlp); 
        
% Display results
Hspa = [Hspa; zeros(3,220)]; % Add three zero rows to the SPA solution for display. 
affichage([Hhot' Hspa' Hx' Hlp'],16,20,11); 
title('From top to bottom: Hottopixx, SPA, XRAY, new LP'); 