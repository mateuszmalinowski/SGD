function [x, f] = gd_matlab(funObj, funPred, x0, valid, options, varargin)
%SGD_MATLAB Gradient descent; matlab implementation.
% We assume the objective is being minimized.
% 
% In:
%   funObj - objective function handler;
%     it returns costs and gradient in this order; precisely
%     [cost, gradient] = funObj(x0, dataIndices, varargin{:})
%     where x0 is a parameter, dataIndices is a set of indices chosen to
%     calculate the (noisy) gradient, and varargin are extra arguments
%   funPred - prediction function;
%     predLabels = funPred(x0, train.examples, varargin{:})
%   x0 - starting point; in parameter's space
%   valid - validation set; We use the validation set to choose the best
%     parameters found so far with respect to the validation set
%     .examples \in R[d,n] where d is dimensionality of
%       datum and n is the number of data points
%     .labels \in Z[n] 
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

if ~isempty(valid)
  validX = valid.examples;
  validY = valid.labels;
  isValidation = true;
  bestValidAcc = 0;
else
  isValidation = false;
end

% different options
nEpochs = options.nEpochs;
numData = options.numData;
if isfield(options, 'MaxIter')
  nIterations = min(options.MaxIter, nEpochs * numData);
else
  nIterations = nEpochs * numData;
end
eta0 = options.eta0;
lambda = options.lambda;
isVerbose = options.isVerbose;
trainX = [];
if isfield(options, 'trainX')
  if ~isempty(options.trainX)
    trainX = options.trainX;
    trainY = options.trainY;
  end
end

x = x0;

it = 1;
while it <= nIterations
  for epochNo = 1:nEpochs
    % in every epoch we re-shuffle data
    dataIndices = randperm(numData);
    
    eta = eta0 / (1 + lambda * eta0 * epochNo);
    fw = 1 - eta * lambda; 
    
    % we pass over all data points
    for k = 1:numData
      
      % computes objective
      [~, grad] = funObj(x, dataIndices(k), varargin{:});
      
      x = fw * x - eta*grad;
      
      it = it + 1;
      
      if it > nIterations
        break;
      end
    end
    
    if isVerbose
      fprintf('Epoch %d\n', epochNo);
      
      if ~isempty(funPred) && ~isempty(trainX)
        trainPred = funPred(x, trainX, varargin{:});
        trainAcc = sum(trainPred == trainY) / length(trainY);
      
        % after each epoch we report current results
        fprintf('-- Accuracy on training set is: %f\n', trainAcc);
      end
      
    end
    
    if isValidation
      validPred = funPred(x, validX);
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

