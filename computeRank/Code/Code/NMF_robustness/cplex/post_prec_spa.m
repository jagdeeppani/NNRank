function K = post_prec_spa(M,~,r)
disp('post_prec_spa started');
[K,~,~] = PostPrecSPA(M,r,0,1,0);
disp('post_prec_spa done');
end