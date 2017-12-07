%function [mat] = gensamples(n)
n = 10000000;
Rho = [1 .5 -0.37; .5 1 -.1; -0.37 -.1 1];
rng('default') % for reproducibility
U = copularnd('Gaussian',Rho,n);
X = [logninv(U(:,1),1.27,0.75)/10 gaminv(U(:,2),2,1)/20 norminv(U(:,3),10)/10];
subplot(1,1,1)
plot3(X(:,1),X(:,2),X(:,3),'.')
grid on
view([-55, 15])
xlabel('X1')
ylabel('X2')
zlabel('X3')
save('sampledata.mat','X')
%end