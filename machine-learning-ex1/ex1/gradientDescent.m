function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
[r no_of_features] = size(X)


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

	hx_arr = zeros(no_of_features,1);	
	summ_arr = zeros(no_of_features,1);
	theta_arr = zeros(no_of_features,1);


	
	summ = 0;
	hx = 0;
	tt0 = 0;
	tt1 = 0;
%{
	for i = 1:m
		hx = theta(1,1) + (theta(2,1) * X(i,2));
		hx = hx - y(i,1);
		%hx = hx * hx;
		summ = summ + hx;
	end
	summ = summ  *  (alpha /m);
	tt0 = theta(1,1) - summ;

	summ = 0;
	for i = 1:m
		hx = theta(1,1) + (theta(2,1) * X(i,2));
		hx = hx - y(i,1);
		hx = hx * X(i,2);
		summ = summ + hx;
	end
	summ = summ  *  (alpha /m);
	tt1 = theta(2,1) - summ;

	theta = [tt0;tt1];
%}
%{

    for k = 1:no_of_features
		for l = 1:m
			hx_arr(k,1) = theta(1,1) + (theta(2,1) * X(l,2));
			hx_arr(k,1) = hx_arr(k,1) - y(l,1);
			hx_arr(k,1) = hx_arr(k,1) * X(l,k);
			summ_arr(k,1) = summ_arr(k,1) + hx_arr(k,1);
		end
		summ_arr(k,1) = summ_arr(k,1)  *  (alpha /m);
		theta_arr(k,1) = theta(k,1) - summ_arr(k,1);
	end

	theta = theta_arr;

%}
 for k = 1:no_of_features
		for l = 1:m
		    f = X(l,:);
			hx_arr(k,1) = theta' * f';
		%	hx_arr(k,1) = theta(1,1) + (theta(2,1) * X(l,2));
			hx_arr(k,1) = hx_arr(k,1) - y(l,1);
			hx_arr(k,1) = hx_arr(k,1) * X(l,k);
			summ_arr(k,1) = summ_arr(k,1) + hx_arr(k,1);
		end
		summ_arr(k,1) = summ_arr(k,1)  *  (alpha /m);
		theta_arr(k,1) = theta(k,1) - summ_arr(k,1);
	end

	theta = theta_arr;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

%end
