function test_label = driver(p,t)

% p = csvread('train.csv',1,2)';
% t = csvread('train.csv',1,1,[1,1,60000,1]);
filter= (1/16)*[-2,1,-2;1,4,1;-2,1,-2];
% filter = rand

input = zeros(784,60000);
for i = 1:60000
    input (:,i) = poslin(reshape(conv2(reshape(p(:,i),28,28),filter,'same'),784,1));
end

layers = 2;
neurons = 100;
alpha = 0.003;
epochs = 20;

disp([layers,neurons,alpha,epochs]);
[W,b,mse] = backprop(input,t,layers,neurons,alpha,epochs,10);

figure;
plot1 = plot([1:epochs],mse);
xlabel('epochs');
ylabel('Mean Squared Error');

test_input = csvread('test.csv',1,1)';
test_label = zeros(10000,2);

x = 0;
for i = 1:length(test_input)
    [m,ind] = max(softmax(W{2}*logsig(W{1}*test_input(:,i)+b{1})+b{2}));
    test_label (i,1) = 60000+i;
    test_label (i,2) = ind - 1;
    %     if(t(i)+1 == ind)
    %         x = x+1;
    %     end
end
% disp(x*100/length(input)); %95.5 percent so far
end