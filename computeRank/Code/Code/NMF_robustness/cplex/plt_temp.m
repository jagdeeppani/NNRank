    funs_str = {
    'SPA',...
    'XRAY',...
    'UTSVD',...
    };


x = 0:1:20;
temp_nmi_val(3,2:21)=temp_nmi_val(3,1:20);
temp_nmi_val(3,1)=0.3658;
plot(x,temp_nmi_val);
grid on;
xlabel='Iterations';
ylabel='Normalized Mutual Information(NMI)';
legend(funs_str{1},funs_str{2},funs_str{3});%,funs_str{4},funs_str{5});