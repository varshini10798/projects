%%
%loading the train and test data 
p=[2,3,4,5];
C_vals=[0.1, 0.6,1.1,2.1];
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
%Implementing SVM for each of the degrees of polynomials and C values that
%have been defined for soft margin
g={};
for m=1:length(p) %iterating through each polynomial degree
    
    p(m)
    
    for n =1:length(C_vals) %iterating through each C value
        g_temp={};
        C_vals(n)
        g_temp{end+1}=p(m);
        g_temp{end+1}=C_vals(n);
        K_nonlinear=((x_train_preprocessed)*transpose(x_train_preprocessed) + 1).^p(m); %obtaining nonlinear kernel
        K_eigenvalues=eig(K_nonlinear);
        flag=0;
        for k=1:length(K_eigenvalues)
            if K_eigenvalues(k)<-1e-6 %checking for Mercer's Condtion
                
                flag=1;
            end
        end
    
        if flag>0
            disp("Kernel not admissible")
            g_temp{end+1}=0;
        else 
            disp("Kernel Admissible")
            g_temp{end+1}=1;
        end
        H=zeros(size(K_nonlinear,1),size(K_nonlinear,1)); %defining dual problem parameters for optimization
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
        C=C_vals(n);
        ub=C*ones(size(K_nonlinear,1),1);
        x0=[];
        alpha=quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options);
        threshold_alpha=1e-4; %defining the threshold to retrieve support vectors
        idx=find(alpha>threshold_alpha & alpha<=C); %finding support vector indices of alpha values that belong to the range (0,C]
        
        wo=zeros(1,size(x_train_preprocessed,2));
        for i=1:size(K_nonlinear,1)
            wo=wo+(alpha(i)*y_train(i)*x_train_preprocessed(i,:));
        end
        
        b=zeros(1,length(idx));
         for i = 1:length(idx)
            sum1=0;
            for j=1:length(y_train)
                sum1=sum1+(alpha(j)*y_train(j)*K_nonlinear(idx(i),j));
            end
    
            b(i)=y_train(idx(i))-sum1;
        end
        
        b0_hm_nonlinear=mean(b);
        wo_nonlinear_hm=wo;
        K_nonlinear_hm=K_nonlinear;
        
        g_temp{end+1}=wo;
        g_temp{end+1}=b0_hm_nonlinear;

        K_nonlinear_test=zeros(size(x_train,1),size(x_test,1)); %using kernel on test data
        for i=1:size(K_nonlinear_test,1)
            for j=1:size(K_nonlinear_test,2)
                K_nonlinear_test(i,j)=((x_train_preprocessed(i,:)*transpose(x_test_preprocessed(j,:)))+1).^p(m);
            end
        end
        
        g_test=zeros(size(x_test,1),1); %obtaining discriminant function values for test data
        for i=1:length(x_test)
            for j=1:size(K_nonlinear_test,1)
                g_test(i)=g_test(i)+((alpha(j)*y_train(j)*K_nonlinear_test(j,i)));
        
            end
            g_test(i)=g_test(i)+b0_hm_nonlinear;
        end
    
        d_test=sign(g_test); %passing discriminant function values through signum function to obtain desired classification outputs
        accuracy=sum(y_test==d_test)/length(d_test); %obtaining test accuracy
    
        g_temp{end+1}=accuracy;
    
        %g{end+1}=g_temp;

        K_nonlinear_train=zeros(size(x_train,1),size(x_train,1)); %using kernel for train data discriminant points
        for i=1:size(K_nonlinear_train,1)
            for j=1:size(K_nonlinear_train,2)
                K_nonlinear_train(i,j)=((x_train_preprocessed(i,:)*transpose(x_train_preprocessed(j,:)))+1).^p(m);
            end
        end
    
        g_train=zeros(size(x_train,1),1);
        for i=1:length(x_train)
            for j=1:size(K_nonlinear_train,2)
                g_train(i)=g_train(i)+((alpha(j)*y_train(j)*K_nonlinear_train(j,i))); %obtaining discriminant function values for train data
        
            end
            g_train(i)=g_train(i)+b0_hm_nonlinear;
        end
        d_train=sign(g_train); %passing discriminant function outputs through signum function to get classification output
%     for i=1:length(g_test)
%         if g_test(i)>0
%             d_test(i)=1;
%         else
%             d_test(i)=-1;
%         end
%     end

        accuracy_train=sum(y_train==d_train)/length(d_train); %calculating train accuracy
    
        g_temp{end+1}=accuracy_train;
    
        g{end+1}=g_temp;
    end
end

