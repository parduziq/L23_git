%% set params

clear
nInputs = 100;
numVals = (nInputs^2-nInputs)/2;
len = 1000;
binSize = 1.0;
nRuns = 1;
%insert RC values sampled from copula
%mat = importdata('sampledata.mat');
%RC = mat(:,2)- 0.05;
%hist(RC, 100)

%insert correlations from RF
%C = importdata('RFC_covariances.mat');


% set pairwise correlation coefficients
%cVals = RC(1:numVals);
%hist(cVals, 100)

 k = 2;
 lambda = 0.5;
 cVals = mat2gray(gamrnd(k,lambda,numVals,1));
 cVals = cVals - 0.1;
%  hist(cVals,100)
 
% set corr matrix
% D = ones(nInputs, 1);
% D = diag(v);
%  for i = 1:nInputs
%      D(i+1 : end, i) = datasample(RC, (nInputs-i));
%  end
% C = D + tril(D, -1)';

% bins = 50;
% r = randi([1 100],1,5);
% figure
% ax1 = subplot(5, 1, 1);
% hist(C(:, r(1)),bins);
% xlim([-0.1 0.8])
% 
% ax2 = subplot(5, 1, 2);
% hist(C(:, r(2)),bins);
% xlim([-0.1 0.8])
% 
% ax3 = subplot(5, 1, 3);
% hist(C(:, r(3)), bins);
% xlim([-0.1 0.8])
% 
% ax4 = subplot(5, 1, 4);
% hist(C(:, r(4)), bins);
% xlim([-0.1 0.8])
% 
% ax5 = subplot(5, 1, 5);
% hist(C(:, r(5)), bins);
% xlim([-0.1 0.8])

C = zeros(nInputs);
C(~tril(ones(size(C)))) = cVals;
C = C + tril(C.',-1);
C(eye(size(C)) ==1) = 1;
figure
imagesc(C)


%% run sampling

mu = ones(1,nInputs) * 0.005; %5Hz
%mu = norminv(mu);
C = C/20;
[S,g,L] = sampleDichGauss01(mu,C,1000,0); %30 samples
figure
plotRaster(logical(S))

save('/Users/qendresa/Desktop/L23/data/matlabData/RC_spiketrains.mat', 'S')



