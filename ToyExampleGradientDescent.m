% Toy example of how sgd_matlab works;
% based on sgd_simple.m from jsgd INRIA
% (http://lear.inrialpes.fr/src/jsgd/)
% 

function ExampleGradientDescent

rng('default')
rng(0);

% dimension of data
d = 2;        

nclass = 5;  % number of classes 
nex = 200;   % number of examples per class (train + test) 

[Xtrain, Ltrain, Xtest, Ltest] = generate_mixture_of_gaussians( ...
  d, nclass, nex, 0.05);

% add bias terms
Xtrain = [Xtrain; ones(1, size(Xtrain, 2))];
Xtest  = [Xtest; ones(1, size(Xtest, 2))];

ntrain = size(Xtrain, 2);
ntest = size(Xtest, 2);

% Graphic output
close all 
plot_with_labels(Xtrain(1:end-1,:), Ltrain);
title('Train data')


% keep some train data for validation
nvalid = floor(ntrain / 5);

Xvalid = Xtrain(:, 1:nvalid);
Lvalid = Ltrain(1:nvalid);
Xtrain = Xtrain(:,nvalid+1:end); 
Ltrain = Ltrain(nvalid+1:end);

% pack train and valid
valid.examples = Xvalid;
valid.labels = Lvalid;

% initialization of the weights
w = zeros((d+1) * nclass, 1);

% training parameters
opt = struct(); 
opt.lambda = 1e-4;
opt.eta0 = 1.0;
opt.nEpochs = 50;
opt.isVerbose = true;
opt.numData = size(Xtrain, 2);

% objective function and the prediction function for SVM
SVM_C = 0.00001;
funObj = @(w, indices) noisy_softmax_cost(w, indices, SVM_C, ...
  nclass, Xtrain, Ltrain);
funPred = @(w, X) softmax_pred(w, X);
% funPred = [];

% first we check if gradients are ok
randTheta = rand(size(w));
J = @(w) funObj(w, randi(size(Xtrain, 2)));
numgrad = compute_numerical_gradient(J, randTheta);
[~, symgrad] = J(randTheta);

% run GD
w = gd_matlab(funObj, funPred, w, valid, opt);
W = reshape(w, nclass, d+1);

fprintf('\n\n');

% evaluate W on training
% [~, predLabels] = max(W * Xtrain, [], 1);
predLabels = softmax_pred(W, Xtrain);
trainAcc = sum(predLabels == Ltrain) / ntrain;
fprintf('Train Accuracy is %f\n', trainAcc);

% evaluate W on test
predLabels  = softmax_pred(W, Xtest);
testAcc = sum(predLabels == Ltest) / ntest;
fprintf('Test Accuracy is %f\n', testAcc);

% graphic output,  
figure
plot_with_labels(Xtest(1:end-1,:), predLabels); 
title('Classified test')

end
