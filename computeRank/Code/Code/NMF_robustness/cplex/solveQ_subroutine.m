function L=solveQ_subroutine(P)
tolerance=0.001;
P=horzcat(P,-P);
[L,~] = MinVolEllipse([P, tolerance);

end
