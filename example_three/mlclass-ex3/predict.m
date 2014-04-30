function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(m,1) X];

% input for Hidden layer
a_first = X;

% weight for hidden layer should be theta1 
z = X*Theta1';

% input for Output layer 

a_second = sigmoid(z);

% modified a_second with extra +1 

a_second = [ ones(m,1) a_second];

% output 

z_out = a_second*Theta2';

%sigmoid 

[idx,p] = max(sigmoid(z_out),[],2);


%p = a_third;







% =========================================================================


end
