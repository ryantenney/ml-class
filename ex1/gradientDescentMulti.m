function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = length(X(1, :));

J_history = zeros(num_iters, 1);

aa = alpha * (1/m);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

	deviation = (X*theta)-y;
	tmp = zeros(n, 1);

	for feature = 1:n
		tmp(feature) =sum(deviation .* X(:, feature));
	end

	theta -= aa * tmp;
	%theta -= aa * [
		%sum(deviation);
		%sum(deviation .* X(:,2))
	%];




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
