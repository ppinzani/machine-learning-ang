data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples
X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;


for iter = 1:iterations

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    H = X * theta;
    delta_0 = (1/m) * sum(((H - y) .* X(:,1))); 
    delta_1 = (1/m) * sum(((H - y) .* X(:,2)));
    theta_0 = theta(1) - alpha * delta_0;
    theta_1 = theta(2) - alpha * delta_1;

    theta = [theta_0;theta_1];
    cost = computeCost(X, y, theta)
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end