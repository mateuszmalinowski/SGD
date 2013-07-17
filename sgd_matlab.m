function [x, f] = sgd_matlab(funObj, funPred, x0, train, valid, options, varargin)
%SGD_MATLAB Stochastic gradient descent; matlab implementation.
%  It is extreme implementation of SGD, meaning it considers only one
%  example to compute gradient. We assume the objective is minimized.
% 
% In:
%   funObj - objective function handler;
%     it returns costs and gradient in this order; precisely
%     [cost, gradient] = funObj(x0, train.examples, train.labels, varargin{:});
%   funPred - prediction function;
%     predLabels = funPred(x0, train.examples, varargin{:})
%   x0 - starting point; in parameter's space
%   train - training set containing:
%     .examples \in R[d,n] where d is dimensionality of
%       datum and n is the number of data points
%     .labels \in Z[n] 
%   valid - validation set;
%     the same structure as train; if valid=[] then validation is absent
%     and the best parameter is taken from the last update or the one that
%     exhibits the lowest cost (if options.takeLowest is true)
%   options - additional optimization settings;
%     defaults are used for non-existent or blank fields
%   varargin - additional arguments to the objective function [optional]
% 
% Out:
%   x - minimum value found
%   f - fuction value at the minimum found
% 
% Mateusz Malinowski
% mmalinow@mpi-inf.mpg.de
% 

trainX = train.examples;
trainY = train.labels;

if ~isempty(valid)
  validX = valid.examples;
  validY = valid.labels;
  isValidation = true;
  bestValidAcc = 0;
else
  isValidation = false;
end

numData = size(trainX, 2);
% numFeatures = size(trainX, 1);
% numClasses = length(unique(trainY));

% different options
nEpochs = options.nEpochs;
if isfield(options, 'MaxIter')
  nIterations = min(options.MaxIter, nEpochs * numData);
else
  nIterations = nEpochs * numData;
end
eta0 = options.eta0;
lambda = options.lambda;
isVerbose = options.isVerbose;

x = x0;

it = 1;
while it <= nIterations
  for epochNo = 1:nEpochs
    % in every epoch we re-shuffle data
    dataIndices = randperm(numData);
    Xshuffle = trainX(:, dataIndices);
    Lshuffle = trainY(dataIndices);
    
    eta = eta0 / (1 + lambda * eta0 * epochNo);
    fw = 1 - eta * lambda; 
    
    % we pass over all data points
    for k = 1:numData
      kthExample = Xshuffle(:, k);
      kthLabel = Lshuffle(k);
      
      % computes objective
      [~, grad] = funObj(x, kthExample, kthLabel, varargin{:});
      
      x = fw * x - eta*grad;
      
      it = it + 1;
      
      if it > nIterations
        break;
      end
    end
    
    if isVerbose
      fprintf('Epoch %d\n', epochNo);
      
      if ~isempty(funPred)
        trainPred = funPred(x, trainX, varargin{:});
        trainAcc = sum(trainPred == trainY) / length(trainY);
      
        % after each epoch we report current results
        fprintf('-- Accuracy on training set is: %f\n', trainAcc);
      end
      
    end
    
    if isValidation
      validPred = funPred(x, validX, varargin{:});
      validAcc = sum(validPred == validY) / length(validY);
      
      if validAcc > bestValidAcc
        bestValidAcc = validAcc;
        bestX = x;
      end
      
      if isVerbose
        fprintf('-- Accuracy on validation set is: %f\n', validAcc);
        fprintf('-- Best accuracy on validation set: %f\n', bestValidAcc);
      end
    end   
    
  end
  
end
  
if isValidation
  x = bestX;
end

if nargout == 2
  [~, f] = funObj(x, trainX, trainY, varargin{:});
end

end

