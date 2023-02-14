%%
%loading train and test data
train_data=load('C:\Users\Varshini\Downloads\SVM_ProjectDescription_24032022\SVM_ProjectDescription_24032022\train.mat');
test_data=load('C:\Users\Varshini\Downloads\SVM_ProjectDescription_24032022\SVM_ProjectDescription_24032022\test.mat');
x_train=transpose(train_data.train_data);
y_train=train_data.train_label;
x_test=transpose(test_data.test_data);
y_test=test_data.test_label;
x_train_preprocessed=zscore(x_train); %standardizing the train data

mean_train=mean(x_train); %obtaining mean of features present in train data
std_train=std(x_train,0,1); %obtainining standard deviation of features in train data
x_test_preprocessed=(x_test-mean_train)./std_train; %standardizing test data
%%
p=[2,3,4,5]; %defining list of degrees of polynomial
g_nonlinear_hard={};
for m=1:length(p) %iterating through each degree of polynomial
    p(m)
    g_temp={};
    g_temp{end+1}=p(m);
    K_nonlinear=(x_train_preprocessed*transpose(x_train_preprocessed) + 1).^p(m); %defining the non linar kernel
    K_eigenvalues=eig(K_nonlinear); %obtaining eigenvalues of nonlinear kernel
    sprintf("min=%d",min(K_eigenvalues))
    flag=0;
    for n=1:length(K_eigenvalues) 
        if K_eigenvalues(n)<-1e-6 %checking Mercer's condition
            
            flag=1;
        end
    end

    if flag>0
        disp("Kernel not admissible")
    else 
        disp("Kernel Admissible")
    end
%Defining the dual problem parameters for optimization 
    H=zeros(size(K_nonlinear,1),size(K_nonlinear,1));
    for i=1:size(K_nonlinear,1)
        for j=1:size(K_nonlinear,1)
            H(i,j)=y_train(i)*y_train(j)*K_nonlinear(i,j);
        end
    end
    options=optimset('LargeScale','off','MaxIter',1000);
    f=-1*ones(1,size(K_nonlinear,1));
    A=[];
    b=[];
    Aeq=transpose(y_train);
    beq=0;
    lb=zeros(size(K_nonlinear,1),1);
    C=10^6; %defining C constraint parameter to very high value to indicate hard margin
    ub=C*ones(size(K_nonlinear,1),1);
    x0=[];
    alpha=quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options); %optimizing the problem through quadratic programming function
    threshold_alpha=1e-4; %setting threshold to obtain support vectors
    idx=find(alpha>threshold_alpha); %finding indices corresponding to support vectors
    
    wo=zeros(1,size(x_train_preprocessed,2));
    for i=1:size(K_nonlinear,1)
        wo=wo+(alpha(i)*y_train(i)*x_train_preprocessed(i,:));
    end
    
    y_sv=y_train(idx);
    b=zeros(1,length(y_sv)); %obtaining bias values using support vectors
    for i = 1:length(y_sv)
        sum1=0;
        for j=1:length(y_train)
            sum1=sum1+(alpha(j)*y_train(j)*K_nonlinear(idx(i),j));
        end

        b(i)=y_sv(i)-sum1;
    end
%     b_support=b(idx);
    b0_hm_nonlinear=mean(b); %setting bias to mean of bias values obtained
    wo_nonlinear_hm=wo;
    K_nonlinear_hm=K_nonlinear;
    g_temp{end+1}=wo;
    g_temp{end+1}=b0_hm_nonlinear;

    K_nonlinear_test=zeros(size(x_test,1),size(x_train,1)); %using non linear kernel for test data to obtain discriminant function points
    for i=1:size(K_nonlinear_test,1)
        for j=1:size(K_nonlinear_test,2)
            K_nonlinear_test(i,j)=((x_test_preprocessed(i,:)*transpose(x_train_preprocessed(j,:)))+1).^p(m);
        end
    end
    
    g_test=zeros(size(x_test,1),1);
    for i=1:length(x_test)
        for j=1:size(K_nonlinear_test,2)
            g_test(i)=g_test(i)+((alpha(j)*y_train(j)*K_nonlinear_test(i,j))); %obtaining discriminant function values for test data
    
        end
        g_test(i)=g_test(i)+b0_hm_nonlinear; 
    end
    d_test=sign(g_test); %passing discriminant function values of test data through signum function for getting desired classification outputs
%     for i=1:length(g_test)
%         if g_test(i)>0
%             d_test(i)=1;
%         else
%             d_test(i)=-1;
%         end
%     end

    accuracy=sum(y_test==d_test)/length(d_test); %obtaining the test accuracy

    g_temp{end+1}=accuracy;



    K_nonlinear_train=zeros(size(x_train,1),size(x_train,1)); %using non linear kernel for train data
    for i=1:size(K_nonlinear_train,1)
        for j=1:size(K_nonlinear_train,2)
            K_nonlinear_train(i,j)=((x_train_preprocessed(i,:)*transpose(x_train_preprocessed(j,:)))+1).^p(m);
        end
    end

    g_train=zeros(size(x_train,1),1);
    for i=1:length(x_train)
        for j=1:size(K_nonlinear_train,2)
            g_train(i)=g_train(i)+((alpha(j)*y_train(j)*K_nonlinear_train(i,j))); %obtaining discriminant function points for train inputs
    
        end
        g_train(i)=g_train(i)+b0_hm_nonlinear;
    end
    d_train=sign(g_train); %passing the discriminant function points through signum function to get desired classification outputs
%     for i=1:length(g_test)
%         if g_test(i)>0
%             d_test(i)=1;
%         else
%             d_test(i)=-1;
%         end
%     end

    accuracy_train=sum(y_train==d_train)/length(d_train); %obtaining train accuracy

    g_temp{end+1}=accuracy_train;

    g_nonlinear_hard{end+1}=g_temp;


    
end