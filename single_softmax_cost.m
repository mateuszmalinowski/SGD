function [cost, grad] = single_softmax_cost(theta, x, y, lambda)
% Takes a parameters theta, data point x and its label y and computes cost
% and gradient of softmax classifier.
% 
% In:
%   theta - parameter;
%     theta \in R[numClasses, numFeatures]
%   x - data point
%   y - label of x
%   lambda - weight decay parameter
%
% Out:
%   cost - cost of the softmax regression at a given point theta
%   grad - gradient of the softmax regression at a given point theta
%   
% Written by: Mateusz Malinowski
% Email: mmalinow@mpi-inf.mpg.de
%  

nfeatures = length(x);

% unroll the parameters from theta
theta = reshape(theta, [], nfeatures);
% take parameter corresponding to current class
classTheta = theta(y, :);

% compute the probability matrix probMatrix(r,c) = p(y^c=r | x^c; theta);
thetaTimesX = theta*x;
sumexptheta = sum(exp(thetaTimesX - classTheta * x));
probMatrix = 1.0 / sumexptheta;
regularizationTerm = 0.5 * lambda *  norm(theta(:), 2)^2;
dataTerm = -log(probMatrix);

cost = dataTerm + regularizationTerm;

if nargin == 1
  return;
else

  % prob matrix for all classes
  mymaxi = max(thetaTimesX);
  sumexptheta = sum(exp(thetaTimesX - mymaxi));
  probMatrix = exp(thetaTimesX - mymaxi) / sumexptheta;
  
  thetagrad = -probMatrix;
  thetagrad(y) = 1 - probMatrix(y);
  thetagrad = -thetagrad * x' + lambda * theta;

  % roll the gradient matrices into a vector for minFunc
  grad = thetagrad(:);
  
end

end
