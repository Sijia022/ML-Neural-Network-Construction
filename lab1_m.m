%% ACML Homework: Neural Networks
% 
% Before reading the code
% The code has been seperated into two parts:
% 
% 1. The first part is using the optimal data and parameters to train the model, 
% which can generate the same output as input eight examples. The test results 
% are in 1.4.2;
% 
% 2. The second part includes multiple experiments to test the influence of 
% different input data and hyperparameters, which leads to the final setting of 
% model in the first part.
% 
% 
%--------------------------------------------------------------------------
%% First Part - Train the Model with Optimal Parameter for Testing
% 1.1 Data Generation

%generate learning examples
X = eye(8);

%prepare training data
random_indices = randperm(8, 8)
X_train = X(random_indices, :)
Y_train = X_train;
%add the bias term for input layer
one_col = ones(size(X_train, 1), 1);
X_train_bias = [one_col, X_train];

%--------------------------------------------------------------------------
% 1.2 Parameter Defination

%define the size of three layers
num_layer1 = 8;
num_layer2 = 3;
num_layer3 = 8;

%define hyperparameters learnt from second part
%regularization coefficient
lambda = 0;
%learning rate
alpha = 0.7;
%iteration for model to converge
max_iteration = 8000; 
%threshold for binary output
threshold = 0.5; 
%randomly initalize weight(Theta) between (-t, t)
t = 0.5;
Theta1 = rand(num_layer2, num_layer1+1)*t*2 - t;
Theta2 = rand(num_layer3, num_layer2+1)*t*2 - t;

%--------------------------------------------------------------------------
% 1.3 Model Training

%create a loop to update the weight(Theta) several times, monitor the error
%between model hypothesis and expected output to see when the model converges

%create an empty array for visualization of error(MSE-mean square error)
MSE_all= [];

%--------------------------------------------------------------------------
% 1.3.1 Feedforward

for i = 1:max_iteration

    %from layer 1 to layer 2
    z2 = X_train_bias * Theta1';
    %activations for layer2, sigmoid function is defined the another file
    a2 = sigmoid(z2); 

    %from layer 2 to layer 3
    %add bias term in the hidden layer
    one_col = ones(size(a2, 1), 1); 
    a2_bias = [one_col, a2];

    z3 = a2_bias * Theta2';
    %activations for layer3
    a3 = sigmoid(z3);

%--------------------------------------------------------------------------
% 1.3.2 Backpropagation

    %compute the "error term" in layer3 and layer2
    %layer3
    Detrivate_g_z3 = a3.*(ones(size(a3))-a3);
    s3 = (a3 - Y_train).*Detrivate_g_z3;

    %layer2
    Detrivate_g_z2 = a2.*(ones(size(a2))-a2);
    s2 = (s3*Theta2(:,2:end)).*Detrivate_g_z2;
    
    %compute partial derivatives for update the weight(Theta)
    %set Delta and D_gradient to zero with the same size as the weight(Theta)
    Delta2 = zeros(num_layer3, num_layer2+1);
    Delta1 = zeros(num_layer2, num_layer1+1);
    D_gradient2 = zeros(num_layer3, num_layer2+1);
    D_gradient1 = zeros(num_layer2, num_layer1+1);

    %from layer3 to layer2
    Delta2(:,2:end) = Delta2(:,2:end) + s3'*a2;
    Delta2(:,1) = Delta2(:,1) + (s3(1,:))';
    %from layer2 to layer1
    Delta1(:,2:end) = Delta1(:,2:end) + s2'*X_train;
    Delta1(:,1) = Delta1(:,1) + (s2(1,:))';
    
    %add regularization term to partial derivatives, but not on the bias term 
    %from layer3 to layer2
    D_gradient2(:,2:end) = (1/size(X_train,1))*(Delta2(:,2:end) + (lambda*Theta2(:,2:end)));
    D_gradient2(:,1) = (1/size(X_train,1))*Delta2(:,1);
    %from layer2 to layer1
    D_gradient1(:,2:end) = (1/size(X_train,1))*(Delta1(:,2:end) + (lambda*Theta1(:,2:end)));
    D_gradient1(:,1) = (1/size(X_train,1))*Delta1(:,1);
    
    %update the weight(Theta) between layers
    Theta2 = Theta2 - alpha*D_gradient2;
    Theta1 = Theta1 - alpha*D_gradient1;

%--------------------------------------------------------------------------
% 1.3.3 Monitor Model Performance

    %moniter the error between hypothesis and expected output every 100 iterations
    %to check if the model converges using MSE(mean square error)

    evaluation_interval = 100;
    if mod(i, evaluation_interval) == 0

        %compute MSE
        MSE = (1/size(Y_train, 1))*sum(sum((a3-Y_train).^2));

        %store the MSE for plot later
        MSE_all = [MSE_all, MSE];

        fprintf('Iteration %d: MSE = %.4f\n', i, MSE);
        disp(' ');
    end
end

%plot the tendency of MSE
iterations = 1:evaluation_interval:max_iteration; %set X

figure;
plot(iterations, MSE_all, 'b', 'DisplayName', 'Training MSE');
title('Training MSE');
xlabel('Iterations');
ylabel('MSE');
legend;
grid on;

%--------------------------------------------------------------------------
% 1.4 Data Testing
% 1.4.1 Feedforward

%test on the same dataset using optimal weights
%add bias term

random_indices_test = randperm(8, 8)
X_test = X(random_indices_test, :)

one_col = ones(size(X_test, 1), 1);
X_test_bias = [one_col, X_test];

%feedforward
%layer1 to layer2
z2 = X_test_bias * Theta1';
a2 = sigmoid(z2);

%layer2 to layer3
%add bias term
one_col = ones(size(a2, 1), 1);
a2_bias = [one_col, a2];

z3 = a2_bias * Theta2';
a3 = sigmoid(z3);

%--------------------------------------------------------------------------
% 1.4.2 Output Generation

%convert probability into binary results
output_labels = a3 >= threshold

%measure the accuracy
sum_total = size(X_test, 1);
m = all(X_test == output_labels, 2);
rowMatches = sum(m)
accuracy = rowMatches/sum_total

%--------------------------------------------------------------------------
% 1.4.3 Final weights and activations of the model

%display final learnt weight and activations
disp('a3 =');
disp(a3);

disp('a2 =');
disp(a2);

disp('Theta2 =');
disp(Theta2)

disp('Theta1 =');
disp(Theta1)

%plot weights(Theta1) to see the relationship
Theta1_1 = sort(Theta1(1,2:end));
Theta1_2 = sort(Theta1(2,2:end));
Theta1_3 = sort(Theta1(3,2:end));
plot(Theta1_1,'DisplayName', 'Weights to get node1 in layer2')
hold on
plot(Theta1_2,'DisplayName', 'Weights to get node2 in layer2')
plot(Theta1_3,'DisplayName', 'Weights to get node3 in layer2')

xlabel('Index');
ylabel('Value of weights');
title('Sorted Weight Values of Theta1');

grid on;
legend('show')
hold off

%plot weights(Theta2) to see the relationship
Theta2_1 = sort(Theta2(:,2));
Theta2_2 = sort(Theta2(:,3));
Theta2_3 = sort(Theta2(:,4));
plot(Theta2_1,'DisplayName', 'Weights for node1 in layer2')
hold on
plot(Theta2_2,'DisplayName', 'Weights for node2 in layer2')
plot(Theta2_3,'DisplayName', 'Weights for node3 in layer2')

xlabel('Index');
ylabel('Value of weights');
title('Sorted Weight Values of Theta2');

grid on;
legend('show')
hold off
%--------------------------------------------------------------------------
%% Second Part - Test the Influence of Input Data and Hyperparameters
% 2.1 Parameter Testing
% 2.1.1 Regularization coefficient : 
% lambda = 0.1

%regularization coefficient
lambda = 0.1;

%randomly initalize weight(Theta) between (-t, t)
t = 0.5;
Theta1 = rand(num_layer2, num_layer1+1)*t*2 - t;
Theta2 = rand(num_layer3, num_layer2+1)*t*2 - t;

%model training
%create an empty array for visualization of error(MSE-mean square error)
MSE_all= [];

for i = 1:max_iteration
    %feedforward
    %from layer 1 to layer 2
    z2 = X_train_bias * Theta1';
    a2 = sigmoid(z2); 

    %from layer 2 to layer 3
    one_col = ones(size(a2, 1), 1); 
    a2_bias = [one_col, a2];
    z3 = a2_bias * Theta2';
    a3 = sigmoid(z3);

    %backpropagation
    %layer3
    Detrivate_g_z3 = a3.*(ones(size(a3))-a3);
    s3 = (a3 - Y_train).*Detrivate_g_z3;
    %layer2
    Detrivate_g_z2 = a2.*(ones(size(a2))-a2);
    s2 = (s3*Theta2(:,2:end)).*Detrivate_g_z2;
    
    %compute partial derivatives for update the weight(Theta)
    Delta2 = zeros(num_layer3, num_layer2+1);
    Delta1 = zeros(num_layer2, num_layer1+1);
    D_gradient2 = zeros(num_layer3, num_layer2+1);
    D_gradient1 = zeros(num_layer2, num_layer1+1);

    %from layer3 to layer2
    Delta2(:,2:end) = Delta2(:,2:end) + s3'*a2;
    Delta2(:,1) = Delta2(:,1) + (s3(1,:))';
    %from layer2 to layer1
    Delta1(:,2:end) = Delta1(:,2:end) + s2'*X_train;
    Delta1(:,1) = Delta1(:,1) + (s2(1,:))';
    
    %add regularization term to partial derivatives, but not on the bias term 
    %from layer3 to layer2
    D_gradient2(:,2:end) = (1/size(X_train,1))*(Delta2(:,2:end) + (lambda*Theta2(:,2:end)));
    D_gradient2(:,1) = (1/size(X_train,1)).*Delta2(:,1);
    %from layer2 to layer1
    D_gradient1(:,2:end) = (1/size(X_train,1))*(Delta1(:,2:end) + (lambda*Theta1(:,2:end)));
    D_gradient1(:,1) = (1/size(X_train,1))*Delta1(:,1);
    
    %update the weight(Theta) between layers
    Theta2 = Theta2 - alpha*D_gradient2;
    Theta1 = Theta1 - alpha*D_gradient1;

    %moniter the error between hypothesis and expected output every 100 iterations
    evaluation_interval = 100;
    if mod(i, evaluation_interval) == 0
        %compute MSE
        MSE = (1/size(Y_train, 1))*sum(sum((a3-Y_train).^2));
        %store the MSE for plot later
        MSE_all = [MSE_all, MSE];
    end
end

%plot the tendency of MSE
iterations = 1:evaluation_interval:max_iteration; %set X
figure;
plot(iterations, MSE_all, 'b', 'DisplayName', 'Training MSE');
title('Training MSE - lambda = 0.1');
xlabel('Iterations');
ylabel('MSE');
legend;
grid on;


%test the examples

%feedforward
%layer1 to layer2
z2 = X_test_bias * Theta1';
a2 = sigmoid(z2);
%layer2 to layer3
one_col = ones(size(a2, 1), 1);
a2_bias = [one_col, a2];
z3 = a2_bias * Theta2';
a3 = sigmoid(z3);

%convert probability into binary results
output_labels = a3 >= threshold

%measure the accuracy
sum_total = size(X_test, 1);
m = all(X_test == output_labels, 2);
rowMatches = sum(m)
accuracy = rowMatches/sum_total

%--------------------------------------------------------------------------
% lambda = 0.01

%regularization coefficient
lambda = 0.01;

%randomly initalize weight(Theta) between (-t, t)
t = 0.5;
Theta1 = rand(num_layer2, num_layer1+1)*t*2 - t;
Theta2 = rand(num_layer3, num_layer2+1)*t*2 - t;

%model training
%create an empty array for visualization of error(MSE-mean square error)
MSE_all= [];

for i = 1:max_iteration
    %feedforward
    %from layer 1 to layer 2
    z2 = X_train_bias * Theta1';
    a2 = sigmoid(z2); 

    %from layer 2 to layer 3
    one_col = ones(size(a2, 1), 1); 
    a2_bias = [one_col, a2];
    z3 = a2_bias * Theta2';
    a3 = sigmoid(z3);

    %backpropagation
    %layer3
    Detrivate_g_z3 = a3.*(ones(size(a3))-a3);
    s3 = (a3 - Y_train).*Detrivate_g_z3;
    %layer2
    Detrivate_g_z2 = a2.*(ones(size(a2))-a2);
    s2 = (s3*Theta2(:,2:end)).*Detrivate_g_z2;
    
    %compute partial derivatives for update the weight(Theta)
    Delta2 = zeros(num_layer3, num_layer2+1);
    Delta1 = zeros(num_layer2, num_layer1+1);
    D_gradient2 = zeros(num_layer3, num_layer2+1);
    D_gradient1 = zeros(num_layer2, num_layer1+1);

    %from layer3 to layer2
    Delta2(:,2:end) = Delta2(:,2:end) + s3'*a2;
    Delta2(:,1) = Delta2(:,1) + (s3(1,:))';
    %from layer2 to layer1
    Delta1(:,2:end) = Delta1(:,2:end) + s2'*X_train;
    Delta1(:,1) = Delta1(:,1) + (s2(1,:))';
    
    %add regularization term to partial derivatives, but not on the bias term 
    %from layer3 to layer2
    D_gradient2(:,2:end) = (1/size(X_train,1))*(Delta2(:,2:end) + (lambda*Theta2(:,2:end)));
    D_gradient2(:,1) = (1/size(X_train,1)).*Delta2(:,1);
    %from layer2 to layer1
    D_gradient1(:,2:end) = (1/size(X_train,1))*(Delta1(:,2:end) + (lambda*Theta1(:,2:end)));
    D_gradient1(:,1) = (1/size(X_train,1))*Delta1(:,1);
    
    %update the weight(Theta) between layers
    Theta2 = Theta2 - alpha*D_gradient2;
    Theta1 = Theta1 - alpha*D_gradient1;

    %moniter the error between hypothesis and expected output every 100 iterations
    evaluation_interval = 100;
    if mod(i, evaluation_interval) == 0
        %compute MSE
        MSE = (1/size(Y_train, 1))*sum(sum((a3-Y_train).^2));
        %store the MSE for plot later
        MSE_all = [MSE_all, MSE];
    end
end

%plot the tendency of MSE
iterations = 1:evaluation_interval:max_iteration; %set X
figure;
plot(iterations, MSE_all, 'b', 'DisplayName', 'Training MSE');
title('Training MSE - lambda = 0.01');
xlabel('Iterations');
ylabel('MSE');
legend;
grid on;


%test the examples

%feedforward
%layer1 to layer2
z2 = X_test_bias * Theta1';
a2 = sigmoid(z2);
%layer2 to layer3
one_col = ones(size(a2, 1), 1);
a2_bias = [one_col, a2];
z3 = a2_bias * Theta2';
a3 = sigmoid(z3);

%convert probability into binary results
output_labels = a3 >= threshold

%measure the accuracy
sum_total = size(X_test, 1);
m = all(X_test == output_labels, 2);
rowMatches = sum(m)
accuracy = rowMatches/sum_total

%--------------------------------------------------------------------------
% 2.1.2 Learning rate 
% alpha = 0.1

%learning rate
alpha = 0.1;

%reset the optimal regularization coefficient
lambda = 0;

%randomly initalize weight(Theta) between (-t, t)
t = 0.5;
Theta1 = rand(num_layer2, num_layer1+1)*t*2 - t;
Theta2 = rand(num_layer3, num_layer2+1)*t*2 - t;

%model training
%create an empty array for visualization of error(MSE-mean square error)
MSE_all= [];

for i = 1:max_iteration
    %feedforward
    %from layer 1 to layer 2
    z2 = X_train_bias * Theta1';
    a2 = sigmoid(z2); 

    %from layer 2 to layer 3
    one_col = ones(size(a2, 1), 1); 
    a2_bias = [one_col, a2];
    z3 = a2_bias * Theta2';
    a3 = sigmoid(z3);

    %backpropagation
    %layer3
    Detrivate_g_z3 = a3.*(ones(size(a3))-a3);
    s3 = (a3 - Y_train).*Detrivate_g_z3;
    %layer2
    Detrivate_g_z2 = a2.*(ones(size(a2))-a2);
    s2 = (s3*Theta2(:,2:end)).*Detrivate_g_z2;
    
    %compute partial derivatives for update the weight(Theta)
    Delta2 = zeros(num_layer3, num_layer2+1);
    Delta1 = zeros(num_layer2, num_layer1+1);
    D_gradient2 = zeros(num_layer3, num_layer2+1);
    D_gradient1 = zeros(num_layer2, num_layer1+1);

    %from layer3 to layer2
    Delta2(:,2:end) = Delta2(:,2:end) + s3'*a2;
    Delta2(:,1) = Delta2(:,1) + (s3(1,:))';
    %from layer2 to layer1
    Delta1(:,2:end) = Delta1(:,2:end) + s2'*X_train;
    Delta1(:,1) = Delta1(:,1) + (s2(1,:))';
    
    %add regularization term to partial derivatives, but not on the bias term 
    %from layer3 to layer2
    D_gradient2(:,2:end) = (1/size(X_train,1))*(Delta2(:,2:end) + (lambda*Theta2(:,2:end)));
    D_gradient2(:,1) = (1/size(X_train,1)).*Delta2(:,1);
    %from layer2 to layer1
    D_gradient1(:,2:end) = (1/size(X_train,1))*(Delta1(:,2:end) + (lambda*Theta1(:,2:end)));
    D_gradient1(:,1) = (1/size(X_train,1))*Delta1(:,1);
    
    %update the weight(Theta) between layers
    Theta2 = Theta2 - alpha*D_gradient2;
    Theta1 = Theta1 - alpha*D_gradient1;

    %moniter the error between hypothesis and expected output every 100 iterations
    evaluation_interval = 100;
    if mod(i, evaluation_interval) == 0
        %compute MSE
        MSE = (1/size(Y_train, 1))*sum(sum((a3-Y_train).^2));
        %store the MSE for plot later
        MSE_all = [MSE_all, MSE];
    end
end

%plot the tendency of MSE
iterations = 1:evaluation_interval:max_iteration; %set X
figure;
plot(iterations, MSE_all, 'b', 'DisplayName', 'Training MSE');
title('Training MSE - alpha = 0.1');
xlabel('Iterations');
ylabel('MSE');
legend;
grid on;


%test the examples

%feedforward
%layer1 to layer2
z2 = X_test_bias * Theta1';
a2 = sigmoid(z2);
%layer2 to layer3
one_col = ones(size(a2, 1), 1);
a2_bias = [one_col, a2];
z3 = a2_bias * Theta2';
a3 = sigmoid(z3);

%convert probability into binary results
output_labels = a3 >= threshold

%measure the accuracy
sum_total = size(X_test, 1);
m = all(X_test == output_labels, 2);
rowMatches = sum(m)
accuracy = rowMatches/sum_total

%--------------------------------------------------------------------------
% alpha = 0.9

%learning rate
alpha = 0.9;

%randomly initalize weight(Theta) between (-t, t)
t = 0.5;
Theta1 = rand(num_layer2, num_layer1+1)*t*2 - t;
Theta2 = rand(num_layer3, num_layer2+1)*t*2 - t;

%model training
%create an empty array for visualization of error(MSE-mean square error)
MSE_all= [];

for i = 1:max_iteration
    %feedforward
    %from layer 1 to layer 2
    z2 = X_train_bias * Theta1';
    a2 = sigmoid(z2); 

    %from layer 2 to layer 3
    one_col = ones(size(a2, 1), 1); 
    a2_bias = [one_col, a2];
    z3 = a2_bias * Theta2';
    a3 = sigmoid(z3);

    %backpropagation
    %layer3
    Detrivate_g_z3 = a3.*(ones(size(a3))-a3);
    s3 = (a3 - Y_train).*Detrivate_g_z3;
    %layer2
    Detrivate_g_z2 = a2.*(ones(size(a2))-a2);
    s2 = (s3*Theta2(:,2:end)).*Detrivate_g_z2;
    
    %compute partial derivatives for update the weight(Theta)
    Delta2 = zeros(num_layer3, num_layer2+1);
    Delta1 = zeros(num_layer2, num_layer1+1);
    D_gradient2 = zeros(num_layer3, num_layer2+1);
    D_gradient1 = zeros(num_layer2, num_layer1+1);

    %from layer3 to layer2
    Delta2(:,2:end) = Delta2(:,2:end) + s3'*a2;
    Delta2(:,1) = Delta2(:,1) + (s3(1,:))';
    %from layer2 to layer1
    Delta1(:,2:end) = Delta1(:,2:end) + s2'*X_train;
    Delta1(:,1) = Delta1(:,1) + (s2(1,:))';
    
    %add regularization term to partial derivatives, but not on the bias term 
    %from layer3 to layer2
    D_gradient2(:,2:end) = (1/size(X_train,1))*(Delta2(:,2:end) + (lambda*Theta2(:,2:end)));
    D_gradient2(:,1) = (1/size(X_train,1)).*Delta2(:,1);
    %from layer2 to layer1
    D_gradient1(:,2:end) = (1/size(X_train,1))*(Delta1(:,2:end) + (lambda*Theta1(:,2:end)));
    D_gradient1(:,1) = (1/size(X_train,1))*Delta1(:,1);
    
    %update the weight(Theta) between layers
    Theta2 = Theta2 - alpha*D_gradient2;
    Theta1 = Theta1 - alpha*D_gradient1;

    %moniter the error between hypothesis and expected output every 100 iterations
    evaluation_interval = 100;
    if mod(i, evaluation_interval) == 0
        %compute MSE
        MSE = (1/size(Y_train, 1))*sum(sum((a3-Y_train).^2));
        %store the MSE for plot later
        MSE_all = [MSE_all, MSE];
    end
end

%plot the tendency of MSE
iterations = 1:evaluation_interval:max_iteration; %set X
figure;
plot(iterations, MSE_all, 'b', 'DisplayName', 'Training MSE');
title('Training MSE - alpha = 0.9');
xlabel('Iterations');
ylabel('MSE');
legend;
grid on;


%test the examples

%feedforward
%layer1 to layer2
z2 = X_test_bias * Theta1';
a2 = sigmoid(z2);
%layer2 to layer3
one_col = ones(size(a2, 1), 1);
a2_bias = [one_col, a2];
z3 = a2_bias * Theta2';
a3 = sigmoid(z3);

%convert probability into binary results
output_labels = a3 >= threshold

%measure the accuracy
sum_total = size(X_test, 1);
m = all(X_test == output_labels, 2);
rowMatches = sum(m)
accuracy = rowMatches/sum_total

%--------------------------------------------------------------------------
% 2.1.3 Numbers of iteration (batch gradient descent)
% max_iteration = 1000

%iteration for model to converge
max_iteration = 1000; 

%reset learning rate
alpha = 0.5;

%randomly initalize weight(Theta) between (-t, t)
t = 0.5;
Theta1 = rand(num_layer2, num_layer1+1)*t*2 - t;
Theta2 = rand(num_layer3, num_layer2+1)*t*2 - t;

%model training
%create an empty array for visualization of error(MSE-mean square error)
MSE_all= [];

for i = 1:max_iteration
    %feedforward
    %from layer 1 to layer 2
    z2 = X_train_bias * Theta1';
    a2 = sigmoid(z2); 

    %from layer 2 to layer 3
    one_col = ones(size(a2, 1), 1); 
    a2_bias = [one_col, a2];
    z3 = a2_bias * Theta2';
    a3 = sigmoid(z3);

    %backpropagation
    %layer3
    Detrivate_g_z3 = a3.*(ones(size(a3))-a3);
    s3 = (a3 - Y_train).*Detrivate_g_z3;
    %layer2
    Detrivate_g_z2 = a2.*(ones(size(a2))-a2);
    s2 = (s3*Theta2(:,2:end)).*Detrivate_g_z2;
    
    %compute partial derivatives for update the weight(Theta)
    Delta2 = zeros(num_layer3, num_layer2+1);
    Delta1 = zeros(num_layer2, num_layer1+1);
    D_gradient2 = zeros(num_layer3, num_layer2+1);
    D_gradient1 = zeros(num_layer2, num_layer1+1);

    %from layer3 to layer2
    Delta2(:,2:end) = Delta2(:,2:end) + s3'*a2;
    Delta2(:,1) = Delta2(:,1) + (s3(1,:))';
    %from layer2 to layer1
    Delta1(:,2:end) = Delta1(:,2:end) + s2'*X_train;
    Delta1(:,1) = Delta1(:,1) + (s2(1,:))';
    
    %add regularization term to partial derivatives, but not on the bias term 
    %from layer3 to layer2
    D_gradient2(:,2:end) = (1/size(X_train,1))*(Delta2(:,2:end) + (lambda*Theta2(:,2:end)));
    D_gradient2(:,1) = (1/size(X_train,1)).*Delta2(:,1);
    %from layer2 to layer1
    D_gradient1(:,2:end) = (1/size(X_train,1))*(Delta1(:,2:end) + (lambda*Theta1(:,2:end)));
    D_gradient1(:,1) = (1/size(X_train,1))*Delta1(:,1);
    
    %update the weight(Theta) between layers
    Theta2 = Theta2 - alpha*D_gradient2;
    Theta1 = Theta1 - alpha*D_gradient1;

    %moniter the error between hypothesis and expected output every 100 iterations
    evaluation_interval = 100;
    if mod(i, evaluation_interval) == 0
        %compute MSE
        MSE = (1/size(Y_train, 1))*sum(sum((a3-Y_train).^2));
        %store the MSE for plot later
        MSE_all = [MSE_all, MSE];
    end
end

%plot the tendency of MSE
iterations = 1:evaluation_interval:max_iteration; %set X
figure;
plot(iterations, MSE_all, 'b', 'DisplayName', 'Training MSE');
title('Training MSE - iteration_max = 1000');
xlabel('Iterations');
ylabel('MSE');
legend;
grid on;


%test the examples

%feedforward
%layer1 to layer2
z2 = X_test_bias * Theta1';
a2 = sigmoid(z2);
%layer2 to layer3
one_col = ones(size(a2, 1), 1);
a2_bias = [one_col, a2];
z3 = a2_bias * Theta2';
a3 = sigmoid(z3);

%convert probability into binary results
output_labels = a3 >= threshold

%measure the accuracy
sum_total = size(X_test, 1);
m = all(X_test == output_labels, 2);
rowMatches = sum(m)
accuracy = rowMatches/sum_total

%--------------------------------------------------------------------------
% max_iteration = 5000

%iteration for model to converge
max_iteration = 5000; 

%randomly initalize weight(Theta) between (-t, t)
t = 0.5;
Theta1 = rand(num_layer2, num_layer1+1)*t*2 - t;
Theta2 = rand(num_layer3, num_layer2+1)*t*2 - t;

%model training
%create an empty array for visualization of error(MSE-mean square error)
MSE_all= [];

for i = 1:max_iteration
    %feedforward
    %from layer 1 to layer 2
    z2 = X_train_bias * Theta1';
    a2 = sigmoid(z2); 

    %from layer 2 to layer 3
    one_col = ones(size(a2, 1), 1); 
    a2_bias = [one_col, a2];
    z3 = a2_bias * Theta2';
    a3 = sigmoid(z3);

    %backpropagation
    %layer3
    Detrivate_g_z3 = a3.*(ones(size(a3))-a3);
    s3 = (a3 - Y_train).*Detrivate_g_z3;
    %layer2
    Detrivate_g_z2 = a2.*(ones(size(a2))-a2);
    s2 = (s3*Theta2(:,2:end)).*Detrivate_g_z2;
    
    %compute partial derivatives for update the weight(Theta)
    Delta2 = zeros(num_layer3, num_layer2+1);
    Delta1 = zeros(num_layer2, num_layer1+1);
    D_gradient2 = zeros(num_layer3, num_layer2+1);
    D_gradient1 = zeros(num_layer2, num_layer1+1);

    %from layer3 to layer2
    Delta2(:,2:end) = Delta2(:,2:end) + s3'*a2;
    Delta2(:,1) = Delta2(:,1) + (s3(1,:))';
    %from layer2 to layer1
    Delta1(:,2:end) = Delta1(:,2:end) + s2'*X_train;
    Delta1(:,1) = Delta1(:,1) + (s2(1,:))';
    
    %add regularization term to partial derivatives, but not on the bias term 
    %from layer3 to layer2
    D_gradient2(:,2:end) = (1/size(X_train,1))*(Delta2(:,2:end) + (lambda*Theta2(:,2:end)));
    D_gradient2(:,1) = (1/size(X_train,1)).*Delta2(:,1);
    %from layer2 to layer1
    D_gradient1(:,2:end) = (1/size(X_train,1))*(Delta1(:,2:end) + (lambda*Theta1(:,2:end)));
    D_gradient1(:,1) = (1/size(X_train,1))*Delta1(:,1);
    
    %update the weight(Theta) between layers
    Theta2 = Theta2 - alpha*D_gradient2;
    Theta1 = Theta1 - alpha*D_gradient1;

    %moniter the error between hypothesis and expected output every 100 iterations
    evaluation_interval = 100;
    if mod(i, evaluation_interval) == 0
        %compute MSE
        MSE = (1/size(Y_train, 1))*sum(sum((a3-Y_train).^2));
        %store the MSE for plot later
        MSE_all = [MSE_all, MSE];
    end
end

%plot the tendency of MSE
iterations = 1:evaluation_interval:max_iteration; %set X
figure;
plot(iterations, MSE_all, 'b', 'DisplayName', 'Training MSE');
title('Training MSE - iteration_max = 5000');
xlabel('Iterations');
ylabel('MSE');
legend;
grid on;


%test the examples

%feedforward
%layer1 to layer2
z2 = X_test_bias * Theta1';
a2 = sigmoid(z2);
%layer2 to layer3
one_col = ones(size(a2, 1), 1);
a2_bias = [one_col, a2];
z3 = a2_bias * Theta2';
a3 = sigmoid(z3);

%convert probability into binary results
output_labels = a3 >= threshold

%measure the accuracy
sum_total = size(X_test, 1);
m = all(X_test == output_labels, 2);
rowMatches = sum(m)
accuracy = rowMatches/sum_total

%--------------------------------------------------------------------------
% 2.1.4 Weights Initalization
% Initialize Theta to zero

Theta1 = zeros(num_layer2, num_layer1+1);
Theta2 = zeros(num_layer3, num_layer2+1);

%reset iteration for model to converge
max_iteration = 8000; 

%model training
%create an empty array for visualization of error(MSE-mean square error)
MSE_all= [];

for i = 1:max_iteration
    %feedforward
    %from layer 1 to layer 2
    z2 = X_train_bias * Theta1';
    a2 = sigmoid(z2); 

    %from layer 2 to layer 3
    one_col = ones(size(a2, 1), 1); 
    a2_bias = [one_col, a2];
    z3 = a2_bias * Theta2';
    a3 = sigmoid(z3);

    %backpropagation
    %layer3
    Detrivate_g_z3 = a3.*(ones(size(a3))-a3);
    s3 = (a3 - Y_train).*Detrivate_g_z3;
    %layer2
    Detrivate_g_z2 = a2.*(ones(size(a2))-a2);
    s2 = (s3*Theta2(:,2:end)).*Detrivate_g_z2;
    
    %compute partial derivatives for update the weight(Theta)
    Delta2 = zeros(num_layer3, num_layer2+1);
    Delta1 = zeros(num_layer2, num_layer1+1);
    D_gradient2 = zeros(num_layer3, num_layer2+1);
    D_gradient1 = zeros(num_layer2, num_layer1+1);

    %from layer3 to layer2
    Delta2(:,2:end) = Delta2(:,2:end) + s3'*a2;
    Delta2(:,1) = Delta2(:,1) + (s3(1,:))';
    %from layer2 to layer1
    Delta1(:,2:end) = Delta1(:,2:end) + s2'*X_train;
    Delta1(:,1) = Delta1(:,1) + (s2(1,:))';
    
    %add regularization term to partial derivatives, but not on the bias term 
    %from layer3 to layer2
    D_gradient2(:,2:end) = (1/size(X_train,1))*(Delta2(:,2:end) + (lambda*Theta2(:,2:end)));
    D_gradient2(:,1) = (1/size(X_train,1)).*Delta2(:,1);
    %from layer2 to layer1
    D_gradient1(:,2:end) = (1/size(X_train,1))*(Delta1(:,2:end) + (lambda*Theta1(:,2:end)));
    D_gradient1(:,1) = (1/size(X_train,1))*Delta1(:,1);
    
    %update the weight(Theta) between layers
    Theta2 = Theta2 - alpha*D_gradient2;
    Theta1 = Theta1 - alpha*D_gradient1;

    %moniter the error between hypothesis and expected output every 100 iterations
    evaluation_interval = 100;
    if mod(i, evaluation_interval) == 0
        %compute MSE
        MSE = (1/size(Y_train, 1))*sum(sum((a3-Y_train).^2));
        %store the MSE for plot later
        MSE_all = [MSE_all, MSE];
    end
end

%plot the tendency of MSE
iterations = 1:evaluation_interval:max_iteration; %set X
figure;
plot(iterations, MSE_all, 'b', 'DisplayName', 'Training MSE');
title('Training MSE - Initialize Theta to zero');
xlabel('Iterations');
ylabel('MSE');
legend;
grid on;


%test the examples

%feedforward
%layer1 to layer2
z2 = X_test_bias * Theta1';
a2 = sigmoid(z2);
%layer2 to layer3
one_col = ones(size(a2, 1), 1);
a2_bias = [one_col, a2];
z3 = a2_bias * Theta2';
a3 = sigmoid(z3);

%convert probability into binary results
output_labels = a3 >= threshold

%measure the accuracy
sum_total = size(X_test, 1);
m = all(X_test == output_labels, 2);
rowMatches = sum(m)
accuracy = rowMatches/sum_total

%--------------------------------------------------------------------------
% Initialize Theta Between (0,1)

Theta1 = rand(num_layer2, num_layer1+1);
Theta2 = rand(num_layer3, num_layer2+1);

%model training
%create an empty array for visualization of error(MSE-mean square error)
MSE_all= [];

for i = 1:max_iteration
    %feedforward
    %from layer 1 to layer 2
    z2 = X_train_bias * Theta1';
    a2 = sigmoid(z2); 

    %from layer 2 to layer 3
    one_col = ones(size(a2, 1), 1); 
    a2_bias = [one_col, a2];
    z3 = a2_bias * Theta2';
    a3 = sigmoid(z3);

    %backpropagation
    %layer3
    Detrivate_g_z3 = a3.*(ones(size(a3))-a3);
    s3 = (a3 - Y_train).*Detrivate_g_z3;
    %layer2
    Detrivate_g_z2 = a2.*(ones(size(a2))-a2);
    s2 = (s3*Theta2(:,2:end)).*Detrivate_g_z2;
    
    %compute partial derivatives for update the weight(Theta)
    Delta2 = zeros(num_layer3, num_layer2+1);
    Delta1 = zeros(num_layer2, num_layer1+1);
    D_gradient2 = zeros(num_layer3, num_layer2+1);
    D_gradient1 = zeros(num_layer2, num_layer1+1);

    %from layer3 to layer2
    Delta2(:,2:end) = Delta2(:,2:end) + s3'*a2;
    Delta2(:,1) = Delta2(:,1) + (s3(1,:))';
    %from layer2 to layer1
    Delta1(:,2:end) = Delta1(:,2:end) + s2'*X_train;
    Delta1(:,1) = Delta1(:,1) + (s2(1,:))';
    
    %add regularization term to partial derivatives, but not on the bias term 
    %from layer3 to layer2
    D_gradient2(:,2:end) = (1/size(X_train,1))*(Delta2(:,2:end) + (lambda*Theta2(:,2:end)));
    D_gradient2(:,1) = (1/size(X_train,1)).*Delta2(:,1);
    %from layer2 to layer1
    D_gradient1(:,2:end) = (1/size(X_train,1))*(Delta1(:,2:end) + (lambda*Theta1(:,2:end)));
    D_gradient1(:,1) = (1/size(X_train,1))*Delta1(:,1);
    
    %update the weight(Theta) between layers
    Theta2 = Theta2 - alpha*D_gradient2;
    Theta1 = Theta1 - alpha*D_gradient1;

    %moniter the error between hypothesis and expected output every 100 iterations
    evaluation_interval = 100;
    if mod(i, evaluation_interval) == 0
        %compute MSE
        MSE = (1/size(Y_train, 1))*sum(sum((a3-Y_train).^2));
        %store the MSE for plot later
        MSE_all = [MSE_all, MSE];
    end
end

%plot the tendency of MSE
iterations = 1:evaluation_interval:max_iteration; %set X
figure;
plot(iterations, MSE_all, 'b', 'DisplayName', 'Training MSE');
title('Training MSE - Initialize Theta Between (0,1)');
xlabel('Iterations');
ylabel('MSE');
legend;
grid on;


%test the examples

%feedforward
%layer1 to layer2
z2 = X_test_bias * Theta1';
a2 = sigmoid(z2);
%layer2 to layer3
one_col = ones(size(a2, 1), 1);
a2_bias = [one_col, a2];
z3 = a2_bias * Theta2';
a3 = sigmoid(z3);

%convert probability into binary results
output_labels = a3 >= threshold

%measure the accuracy
sum_total = size(X_test, 1);
m = all(X_test == output_labels, 2);
rowMatches = sum(m)
accuracy = rowMatches/sum_total

%--------------------------------------------------------------------------
% 2.2 New Data generation
% 2.2.1 Less Than 8 Example

%prepare training data
X_train = X_train(1:7,:)
Y_train = X_train;
%add the bias term for input layer
one_col = ones(size(X_train, 1), 1);
X_train_bias = [one_col, X_train];

%randomly initalize weight(Theta) between (-t, t)
t = 0.5;
Theta1 = rand(num_layer2, num_layer1+1)*t*2 - t;
Theta2 = rand(num_layer3, num_layer2+1)*t*2 - t;

%model training
%create an empty array for visualization of error(MSE-mean square error)
MSE_all= [];

for i = 1:max_iteration
    %feedforward
    %from layer 1 to layer 2
    z2 = X_train_bias * Theta1';
    a2 = sigmoid(z2); 

    %from layer 2 to layer 3
    one_col = ones(size(a2, 1), 1); 
    a2_bias = [one_col, a2];
    z3 = a2_bias * Theta2';
    a3 = sigmoid(z3);

    %backpropagation
    %layer3
    Detrivate_g_z3 = a3.*(ones(size(a3))-a3);
    s3 = (a3 - Y_train).*Detrivate_g_z3;
    %layer2
    Detrivate_g_z2 = a2.*(ones(size(a2))-a2);
    s2 = (s3*Theta2(:,2:end)).*Detrivate_g_z2;
    
    %compute partial derivatives for update the weight(Theta)
    Delta2 = zeros(num_layer3, num_layer2+1);
    Delta1 = zeros(num_layer2, num_layer1+1);
    D_gradient2 = zeros(num_layer3, num_layer2+1);
    D_gradient1 = zeros(num_layer2, num_layer1+1);

    %from layer3 to layer2
    Delta2(:,2:end) = Delta2(:,2:end) + s3'*a2;
    Delta2(:,1) = Delta2(:,1) + (s3(1,:))';
    %from layer2 to layer1
    Delta1(:,2:end) = Delta1(:,2:end) + s2'*X_train;
    Delta1(:,1) = Delta1(:,1) + (s2(1,:))';
    
    %add regularization term to partial derivatives, but not on the bias term 
    %from layer3 to layer2
    D_gradient2(:,2:end) = (1/size(X_train,1))*(Delta2(:,2:end) + (lambda*Theta2(:,2:end)));
    D_gradient2(:,1) = (1/size(X_train,1)).*Delta2(:,1);
    %from layer2 to layer1
    D_gradient1(:,2:end) = (1/size(X_train,1))*(Delta1(:,2:end) + (lambda*Theta1(:,2:end)));
    D_gradient1(:,1) = (1/size(X_train,1))*Delta1(:,1);
    
    %update the weight(Theta) between layers
    Theta2 = Theta2 - alpha*D_gradient2;
    Theta1 = Theta1 - alpha*D_gradient1;

    %moniter the error between hypothesis and expected output every 100 iterations
    evaluation_interval = 100;
    if mod(i, evaluation_interval) == 0
        %compute MSE
        MSE = (1/size(Y_train, 1))*sum(sum((a3-Y_train).^2));
        %store the MSE for plot later
        MSE_all = [MSE_all, MSE];
    end
end

%plot the tendency of MSE
iterations = 1:evaluation_interval:max_iteration; %set X
figure;
plot(iterations, MSE_all, 'b', 'DisplayName', 'Training MSE');
title('Training MSE - less input');
xlabel('Iterations');
ylabel('MSE');
legend;
grid on;


%test the examples

%feedforward
%layer1 to layer2
z2 = X_test_bias * Theta1';
a2 = sigmoid(z2);
%layer2 to layer3
one_col = ones(size(a2, 1), 1);
a2_bias = [one_col, a2];
z3 = a2_bias * Theta2';
a3 = sigmoid(z3);

%convert probability into binary results
output_labels = a3 >= threshold

%measure the accuracy
sum_total = size(X_test, 1);
m = all(X_test == output_labels, 2);
rowMatches = sum(m)
accuracy = rowMatches/sum_total

%--------------------------------------------------------------------------
% 2.2.2 Add More Binary Data

%generate more 1 from given examples 
n = 8;  
X_new = dec2bin(0:2^n-1, n) - '0';

%number of new dataset
total_rows= 2^n;

%generate a random permutation of row indices
permuted_indices = randperm(total_rows);
X_train = X_new(permuted_indices,:)
Y_train = X_train;

%add the bias term for input layer
one_col = ones(size(X_train, 1), 1);
X_train_bias = [one_col, X_train];
%%
%randomly initalize weight(Theta) between (-t, t)
t = 0.5;
Theta1 = rand(num_layer2, num_layer1+1)*t*2 - t;
Theta2 = rand(num_layer3, num_layer2+1)*t*2 - t;

%model training
%create an empty array for visualization of error(MSE-mean square error)
MSE_all= [];

for i = 1:max_iteration
    %feedforward
    %from layer 1 to layer 2
    z2 = X_train_bias * Theta1';
    a2 = sigmoid(z2); 

    %from layer 2 to layer 3
    one_col = ones(size(a2, 1), 1); 
    a2_bias = [one_col, a2];
    z3 = a2_bias * Theta2';
    a3 = sigmoid(z3);

    %backpropagation
    %layer3
    Detrivate_g_z3 = a3.*(ones(size(a3))-a3);
    s3 = (a3 - Y_train).*Detrivate_g_z3;
    %layer2
    Detrivate_g_z2 = a2.*(ones(size(a2))-a2);
    s2 = (s3*Theta2(:,2:end)).*Detrivate_g_z2;
    
    %compute partial derivatives for update the weight(Theta)
    Delta2 = zeros(num_layer3, num_layer2+1);
    Delta1 = zeros(num_layer2, num_layer1+1);
    D_gradient2 = zeros(num_layer3, num_layer2+1);
    D_gradient1 = zeros(num_layer2, num_layer1+1);

    %from layer3 to layer2
    Delta2(:,2:end) = Delta2(:,2:end) + s3'*a2;
    Delta2(:,1) = Delta2(:,1) + (s3(1,:))';
    %from layer2 to layer1
    Delta1(:,2:end) = Delta1(:,2:end) + s2'*X_train;
    Delta1(:,1) = Delta1(:,1) + (s2(1,:))';
    
    %add regularization term to partial derivatives, but not on the bias term 
    %from layer3 to layer2
    D_gradient2(:,2:end) = (1/size(X_train,1))*(Delta2(:,2:end) + (lambda*Theta2(:,2:end)));
    D_gradient2(:,1) = (1/size(X_train,1)).*Delta2(:,1);
    %from layer2 to layer1
    D_gradient1(:,2:end) = (1/size(X_train,1))*(Delta1(:,2:end) + (lambda*Theta1(:,2:end)));
    D_gradient1(:,1) = (1/size(X_train,1))*Delta1(:,1);
    
    %update the weight(Theta) between layers
    Theta2 = Theta2 - alpha*D_gradient2;
    Theta1 = Theta1 - alpha*D_gradient1;

    %moniter the error between hypothesis and expected output every 100 iterations
    evaluation_interval = 100;
    if mod(i, evaluation_interval) == 0
        %compute MSE
        MSE = (1/size(Y_train, 1))*sum(sum((a3-Y_train).^2));
        %store the MSE for plot later
        MSE_all = [MSE_all, MSE];
    end
end

%plot the tendency of MSE
iterations = 1:evaluation_interval:max_iteration; %set X
figure;
plot(iterations, MSE_all, 'b', 'DisplayName', 'Training MSE');
title('Training MSE - more binary input');
xlabel('Iterations');
ylabel('MSE');
legend;
grid on;


%test the examples

%feedforward
%layer1 to layer2
z2 = X_test_bias * Theta1';
a2 = sigmoid(z2);
%layer2 to layer3
one_col = ones(size(a2, 1), 1);
a2_bias = [one_col, a2];
z3 = a2_bias * Theta2';
a3 = sigmoid(z3)

%convert probability into binary results
output_labels = a3 >= threshold

%measure the accuracy
sum_total = size(X_test, 1);
m = all(X_test == output_labels, 2);
rowMatches = sum(m)
accuracy = rowMatches/sum_total