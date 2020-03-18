function test_label = driver(p,t)

% p = csvread('train.csv',1,2)';
% t = csvread('train.csv',1,1,[1,1,60000,1]);
filter= (1/16)*[-2,1,-2;1,4,1;-2,1,-2];

layers = 2;
neurons = 100;
alpha = 0.003;
epochs = 70;

input = zeros(size(p));
for i = 1:60000
    input (:,i) = poslin(reshape(conv2(reshape(p(:,i),28,28),filter,'same'),784,1));
end

% disp([layers,neurons,alpha,epochs]);
[W,b,mse] = backprop(p,t,layers,neurons,alpha,epochs,10);
[W2,b2,mse2] = backprop(input,t,layers,neurons,alpha,epochs,10);

figure;
plot1 = plot([1:epochs],mse);
xlabel('epochs');
ylabel('Mean Squared Error');
title('MSE for Artificial Neural Network');

figure;
plot1 = plot([1:epochs],mse2);
xlabel('epochs');
ylabel('Mean Squared Error');
title('MSE for Convolutional Neural Network');

% test_input = csvread('test.csv',1,1)';
% test_label = zeros(10000,2);
ann = 0;
cnn = 0;
for i = 1:length(input)
    [m,ind1] = max(softmax(W{2}*logsig(W{1}*p(:,i)+b{1})+b{2}));
    [m,ind2] = max(softmax(W2{2}*logsig(W2{1}*input(:,i)+b2{1})+b2{2}));
    if(t(i)+1 == ind1)
        ann = ann+1;
    end
    if(t(i)+1 == ind2)
        cnn = cnn+1;
    end
end
disp(ann*100/length(input))
disp(cnn*100/length(input)); %95.5 percent so far
end