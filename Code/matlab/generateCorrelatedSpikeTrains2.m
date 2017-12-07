%% set params

clear
nInputs = 50;
len = 1000;
binSize = 1.0;
nRuns = 1;

% set pairwise correlation coefficients
k = 2;
lambda = 0.5;
numVals = (nInputs^2-nInputs)/2;
cVals = mat2gray(gamrnd(k,lambda,numVals,1));
cVals = cVals - 0.1;
hist(cVals,100)

% set corr matrix
C = zeros(nInputs);
C(~tril(ones(size(C)))) = cVals;
C = C + tril(C.',-1);
C(eye(size(C)) ==1) = 1;
imagesc(C)


%% run sampling

mu = ones(1,nInputs) *0.005;
C = C/20;
[S,g,L] = sampleDichGauss01(mu,C,1000,0); 
plotRaster(logical(S))

