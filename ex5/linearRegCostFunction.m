function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

ho_x = X * theta;
reg_tmp = theta;
reg_tmp(1) = 0;

J = 1 /(2*m) * ((ho_x - y)' * (ho_x - y)) + lambda/(2*m) * (reg_tmp' * reg_tmp); 
grad = (1/m) * X' * (ho_x - y) + (lambda/m) * reg_tmp;




% =========================================================================

grad = grad(:);

end
