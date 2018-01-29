function [NMI,P,ConditionalEntropy] = normalized_mutual_information(C1,C2)
% C2 is true labels 
K1=max(C1);
K2=max(C2);
N=length(C1);

P = sparse(C1,C2,1,K1,K2)/N;

P2 = sum(P,1);
P1=sum(P,2);

P12 = P1*P2;

p=find(P);

p1=find(P1); 
H1=-sum(P1(p1).*log(P1(p1)));

p2=find(P2);
H2=-sum(P2(p2).*log(P2(p2)));

NMI = sum(P(p).*log(P(p))) - sum(P(p).*log(P12(p)));
ConditionalEntropy = NMI/H2;
%NMI = 2*NMI/(H1+H2);
%NMI = NMI/(sqrt(H1*H2));
NMI = NMI/max(H1, H2);
NMI = full(NMI);