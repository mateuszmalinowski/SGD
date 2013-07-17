SGD

Minimizes function using Stochastic Gradient Descent Algorithm.
Variation of the L. Bottou's SGD and Inria's JSGD.
This version allows to use arbitrary objective function via the following 
interface (similar to Schmidt's minFunc):
sgd(funObj, funPrediction, x0, train, valid, options, varargin)


I provide the source code together with the example (softmax objective function).
