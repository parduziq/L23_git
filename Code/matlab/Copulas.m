% EPSP-RC copula
n = 1000; rho = .5;
Z = mvnrnd([0 0],[1 rho; rho 1],n);
U = normcdf(Z);
X = [gaminv(U(:,1),2,1)/20 logninv(U(:,2),1.27, 0.75)/10]; %RC vs EPSP

% draw the scatter plot of data with histograms
figure
subplot(1,2,1)
scatterhist(X(:,1),X(:,2),'Direction','out')


%EPSP-STP copula
n = 1000; rho = -0.37;
Z = mvnrnd([0 0],[1 rho; rho 1],n);
U = normcdf(Z);
X = [logninv(U(:,1),1.27)/20 norminv(U(:,2),10)/10]; %RC vs EPSP

% draw the scatter plot of data with histograms
figure
subplot(1,2,2)
scatterhist(X(:,1),X(:,2),'Direction','out')

