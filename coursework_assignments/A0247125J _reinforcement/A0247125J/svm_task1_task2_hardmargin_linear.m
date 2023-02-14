clc
clear all
%%
%load train data and test data
train_data=load('C:\Users\Varshini\Downloads\SVM_ProjectDescription_24032022\SVM_ProjectDescription_24032022\train.mat');
test_data=load('C:\Users\Varshini\Downloads\SVM_ProjectDescription_24032022\SVM_ProjectDescription_24032022\test.mat');
x_train=transpose(train_data.train_data);
y_train=train_data.train_label;
x_test=transpose(test_data.test_data);
y_test=test_data.test_label;

x_train_preprocessed=zscore(x_train); %standardizing the data
K_linear=(x_train_preprocessed)*transpose(x_train_preprocessed); %defining the linear kernel
K_eigenvalues=eig(K_linear); %obtaining eigen values of linear kernel
%%
%checking Mercer's condition
flag=0;
for n=1:length(K_eigenvalues)
    if K_eigenvalues(n)<-1e-4 %checking if eigenvalues are non negative
        disp("Kernel not admissible") %if negative values found then kernel not admissible
        flag=1;
    end
end
if flag==0 
    disp("Kernel admissible") %if nonnegative eigen values then kernel admissible
end
%%
%defining dual problem parameters for optimization and finding the support vectors
H=zeros(size(K_linear,1),size(K_linear,1));
for i=1:size(K_linear,1)
    for j=1:size(K_linear,1)
        H(i,j)=y_train(i)*y_train(j)*K_linear(i,j);
    end
end
options=optimset('LargeScale','off','MaxIter',2000);
f=-1*ones(1,size(K_linear,1));
A=[];
b=[];
Aeq=transpose(y_train);
beq=0;
lb=zeros(size(K_linear,1),1);
C=10^6; %setting C value to a very large value thereby indicating hard margin
ub=C*ones(size(K_linear,1),1);
x0=[];
alpha=quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options); %running optimization function
threshold_alpha=1e-4; %defining threshold for alpha
idx=find(alpha>threshold_alpha); %finding support vectors having alpha value greater than the threshold(i.e 0) 

wo=zeros(1,size(x_train_preprocessed,2));
for i=1:size(K_linear,1)
    wo=wo+(alpha(i)*y_train(i)*x_train_preprocessed(i,:)); %obtaining the weights 
end

b=zeros(1,length(idx));
for i = 1:length(idx)
    b(i)=(1/y_train(idx(i)))-(x_train_preprocessed(idx(i),:)*transpose(wo)); %obtaining the bias values using support vectors
end

b0_hm_linear=mean(b); %taking mean of bias values
wo_linear_hm=wo;
K_linear_hm=K_linear;

%g_linear_hm=[wo;b0_hm_linear];

% p=[2,3,4,5];
% g=zeros(2,length(p));
% for m=1:length(p)
%     K_nonlinear=((x_train_preprocessed)*transpose(x_train_preprocessed) + 1).^p(m);
%     H=zeros(size(K_nonlinear,1),size(K_nonlinear,1));
%     for i=1:size(K_nonlinear,1)
%         for j=1:size(K_nonlinear,1)
%             H(i,j)=y_train(i)*y_train(j)*K_nonlinear(i,j);
%         end
%     end
%     options=optimset('LargeScale','off','MaxIter',2000);
%     f=-1*ones(1,size(K_nonlinear,1));
%     A=[];
%     b=[];
%     Aeq=transpose(y_train);
%     beq=0;
%     lb=zeros(size(K_nonlinear,1),1);
%     C=1e6;
%     ub=C*ones(size(K_nonlinear,1),1);
%     x0=[];
%     alpha=quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options);
%     threshold_alpha=1e-4;
%     idx=find(alpha>threshold_alpha);
%     
%     wo=zeros(1,size(x_train_preprocessed,2));
%     for i=1:size(K_nonlinear,1)
%         wo=wo+(alpha(i)*y_train(i)*x_train_preprocessed(i,:));
%     end
%     
%     b=zeros(1,length(idx));
%     for i = 1:length(idx)
%         b(i)=(1/y_train(idx(i)))-(x_train_preprocessed(idx(i),:)*transpose(wo));
%     end
%     
%     b0_hm_nonlinear=mean(b);
%     wo_nonlinear_hm=wo;
%     K_nonlinear_hm=K_nonlinear;
%     
%     g(:,m)=[wo;b0_hm_nonlinear];
% end
%%
%Obtaining discriminant function and using discriminant function to
%classify the test data points
mean_train=mean(x_train); 
std_train=std(x_train,0,1);
x_test_preprocessed=(x_test-mean_train)./std_train; %standardizing test data using mean and standard deviations of train data

K_linear_test=zeros(size(x_train,1),size(x_test,1));
for i=1:size(K_linear_test,1)
    for j=1:size(K_linear_test,2)
        K_linear_test(i,j)=x_train_preprocessed(i,:)*transpose(x_test_preprocessed(j,:));
    end
end

g_test=zeros(size(x_test,1),1);
for i=1:length(x_test)
    for j=1:size(K_linear_test,1)
        g_test(i)=g_test(i)+((alpha(j)*y_train(j)*K_linear_test(j,i))); %obtaining the values of discriminant function using weight and bias

    end
    g_test(i)=g_test(i)+b0_hm_linear;
end

d_test=sign(g_test); %passing the values through signum function for classification
%     for i=1:length(g_test)
%         if g_test(i)>0
%             d_test(i)=1;
%         else
%             d_test(i)=-1;
%         end
%     end

accuracy_test=sum(y_test==d_test)/length(d_test); %obtaining the test accuracy 

K_linear_train=zeros(size(x_train,1),size(x_train,1));
for i=1:size(K_linear_train,1)
    for j=1:size(K_linear_train,2)
        K_linear_train(i,j)=x_train_preprocessed(i,:)*transpose(x_train_preprocessed(j,:));
    end
end

g_train=zeros(size(x_train,1),1); 
for i=1:length(x_train)
    for j=1:size(K_linear_train,1)
        g_train(i)=g_train(i)+((alpha(j)*y_train(j)*K_linear_train(j,i))); %obtaining values using train data points using discriminant function

    end
    g_train(i)=g_train(i)+b0_hm_linear;
end

d_train=sign(g_train); %passing discriminant function values through signum function for classification
%     for i=1:length(g_test)
%         if g_test(i)>0
%             d_test(i)=1;
%         else
%             d_test(i)=-1;
%         end
%     end

accuracy_train=sum(y_train==d_train)/length(d_train); %obtaining train accuracy

