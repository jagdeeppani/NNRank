function plot_full_synthetic_catchwords(X1, YMatrix1)
%CREATEFIGURE(X1, YMATRIX1)
%  X1:  vector of x data
%  YMATRIX1:  matrix of y data

%  Auto-generated by MATLAB on 01-Mar-2015 11:58:46

% Create figure
figure1 = figure;

% Create axes
axes1 = axes('Parent',figure1,'FontWeight','bold','FontSize',16);
box(axes1,'on');
grid(axes1,'on');
hold(axes1,'all');

% Create multiple lines using matrix input to plot
plot1 = plot(X1,YMatrix1,'Parent',axes1,'MarkerSize',8,'LineWidth',2);
set(plot1(1),'Marker','+','Color',[0 1 0],'DisplayName','Heur-SPA');
set(plot1(2),'Marker','o','Color',[0 0 1],'DisplayName','SPA');
set(plot1(3),'Marker','square','Color',[0 1 1],'DisplayName','XRAY');
set(plot1(4),'Marker','d','Color',[0 0 0],'DisplayName','ER-SPA');
set(plot1(5),'Marker','*','Color',[0 0 1],'DisplayName','ER-XRAY');

set(plot1(6),'Marker','<','Color',[1 0 1],'DisplayName','UTSVD');

% Create xlabel
xlabel('Noise level ($\beta$)','Interpreter','latex','FontWeight','bold',...
    'FontSize',24);

% Create ylabel
ylabel('Normalized $\ell_s$ Residual norm','Interpreter','latex','FontWeight','bold',...
    'FontSize',24);

% Create title
title('Normalized $\ell_s$ Residual for $A$ ( c = 10, $\alpha$ = 0.01)',...
    'Interpreter','latex',...
    'FontWeight','bold',...
    'FontSize',30);

% Create legend
legend1 = legend(axes1,'show');
set(legend1,...
    'Position',[0.723261032161556 0.62107904642409 0.177823485415107 0.295169385194475],...
    'FontSize',20);

