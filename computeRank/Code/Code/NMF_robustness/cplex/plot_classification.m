function plot_classification(X1, YMatrix1)
%CREATEFIGURE(X1, YMATRIX1)
%  X1:  vector of x data
%  YMATRIX1:  matrix of y data

%  Auto-generated by MATLAB on 02-Mar-2015 20:24:51

% Create figure
figure1 = figure;

% Create axes
axes1 = axes('Parent',figure1,'FontWeight','bold','FontSize',16);
box(axes1,'on');
grid(axes1,'on');
hold(axes1,'all');

% Create multiple lines using matrix input to plot
plot1 = plot(X1,YMatrix1,'Parent',axes1,'MarkerSize',8,'LineWidth',2);
set(plot1(1),'Marker','<','Color',[1 0 1],'DisplayName','UTSVD');
set(plot1(2),'Marker','o','Color',[1 0 0],'DisplayName','SPA');
set(plot1(3),'Marker','square','Color',[0 0 1],'DisplayName','XRAY');
set(plot1(4),'Marker','+','Color',[0 1 0],'DisplayName','Heur-SPA');
set(plot1(5),'Marker','d','Color',[0 1 1],'DisplayName','random');
set(plot1(6),'Marker','+','Color',[0 0 0],'DisplayName','full-features');
% Create xlabel
xlabel('Running time (seconds) ','Interpreter','latex','FontWeight','bold','FontSize',24);

% Create ylabel
ylabel('Normalized Mutual Information (NMI)','Interpreter','latex',...
    'FontWeight','bold',...
    'FontSize',24);

% Create title
title('Classification performance for Reuters','Interpreter','latex',...
    'FontWeight','bold',...
    'FontSize',30);

% Create legend
legend1 = legend(axes1,'show');
set(legend1,...
    'Position',[0.718866517524236 0.125941028858217 0.179530201342276 0.320734002509411],...
    'FontSize',20);

