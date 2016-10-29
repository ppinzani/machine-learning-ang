function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

y_mat = zeros(size(y,1),num_labels);

for crnt_label=1:num_labels
	y_mat(:,crnt_label) = y == crnt_label;
end


X = [ones(m,1) X];
Theta1_grad = 1/m * Theta1_grad;
a2 = sigmoid(X*Theta1');
a2 = [ones(size(a2,1),1) a2];

ho_x = sigmoid(a2*Theta2');

temp1 = Theta1;
temp2 = Theta2;

temp1(:,1) = 0;
temp2(:,1) = 0;



J = 0;
for i=1:m
	J = J + ( -y_mat(i,:)*log(ho_x(i,:))' - (1-y_mat(i,:))*log(1-ho_x(i,:))' );
end

J = J/m + lambda/(2*m) * ( sum((temp1.^2)(:)) + sum((temp2.^2)(:)));

%Compute the gradient using the backprop algorithm
for ex = 1:m
	crnt_a1 = X(ex,:);
	crnt_z2 = crnt_a1*Theta1';
	crnt_a2 = sigmoid(crnt_z2);
	crnt_a2 = [1 crnt_a2];
	crnt_a3 = sigmoid(crnt_a2*Theta2');
	crnt_dk3 = (crnt_a3 - y_mat(ex,:))';
	crnt_z2 = [1 crnt_z2];
	crnt_dk2 = Theta2' * crnt_dk3 .* sigmoidGradient(crnt_z2)';
	Theta1_grad = Theta1_grad + crnt_dk2(2:end)*crnt_a1;
	Theta2_grad = Theta2_grad + crnt_dk3*crnt_a2;
end

Theta1_grad = 1/m * Theta1_grad + lambda/m * temp1;
Theta2_grad = 1/m * Theta2_grad + lambda/m * temp2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
