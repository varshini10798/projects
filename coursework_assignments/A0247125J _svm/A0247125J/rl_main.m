%%

close all
load('C:\Users\Varshini\Downloads\RL_ProjectDescription\RL_Project\task1.mat') %loading reward function
%kindly uncomment following line of code to load eval.mat
%load('C:\Users\Varshini\Downloads\RL_ProjectDescription\RL_Project\eval.mat') 
Q1=zeros(size(reward));
trial=1;
e=1e-4;
k=1;
s1=1;
df=0.9; %setting discount factor
%ek=100/(100+k);
%alpha_k=ek;
runs=10; %setting maximum number of runs,trials and time steps
trial_max=3000;
k_max=100;
h=0; %initializing variable calculating number of times goal is reached
goals=zeros(runs,1);
opt={};
flag=0;
flag1=0;
%kindly uncomment below line of code to set reward variable as qevalreward
%reward=qevalreward
states=zeros(10,10); %saving the states to a 10*10 matrix that represents the 10*10 grid.
k=1;
for j=1:size(states,2)
    for i=1:size(states,2)
        states(i,j)=k;
        k=k+1;
    end
end
%%
%reward=qevalreward;
%executing Q-learning algorithm
for run=1:runs %iterating to run 10 times
    run
    Q=zeros(size(reward)); %initialize Q matrix
    Q1=zeros(size(reward));
    for trial=1:trial_max %iterating through trials
        Q=Q1;
        %trial
        k=1;
        s=1;
%         ek=0.2*exp((1+5*log(k))./(k.^2)); 
%         alpha_k=ek;
        while s~=100
%             if s==100
%                 goals(run)=1;
%                 %fprintf("breaking")
%                 break
%             end

            ek=acsch(k/5)+0.5; %defining probability function
            %ek=0.2*(acsch(k/100) +10./k);
                        
            alpha_k=ek; %setting learning rate equal to probability factor
            if alpha_k<0.005 %break from loop if learning rate reaches 0.005
                %disp('breaking')
                break;
            end
            [i_idx,j_idx]=find(states==s); %finding indices of 10*10 matrix containing the state values in order to filter out actions if state is a boundary point in the grid
            idx_vals=ones(length(Q1(s,:)),1);
            if i_idx==1
                idx_vals(1)=0; %setting action index to 0 if action taken makes system go out of the grid
            end
            if j_idx==size(states,2)
                idx_vals(2)=0;
            end
            if i_idx==size(states,1)
                idx_vals(3)=0;
            end
            if j_idx==1
                idx_vals(4)=0;
            end
            p=rand;
            idx=find(idx_vals==1); %obtaining the valid actions
            %idx=find(reward(s,:)~=-1);
            q_samples=Q1(s,idx);
            if any(Q1(s,:)) %implementing exploration or exploitation if q values are non zero
                
                if p>=ek %if probability less than probability factor value in current state do exploitation
                    
                    idx_max=find(q_samples==max(q_samples));
                    ak_max_sample=randperm(length(idx_max),1);
                    ak=idx(idx_max(ak_max_sample));
                else %if probability greater than probability factor value in current state do exploration
                    %q_samples=Q1(s,:);
                    idx_rand=find(q_samples~=max(q_samples));
                    %q_rand=q_samples(idx_rand);
                    %if length(idx_rand)~=0

                    ak_sample=randperm(length(idx_rand),1);
                    ak=idx(idx_rand(ak_sample));
%                     else
%                         ak_sample=randperm(length(q_samples),1);
%                         ak=ak_sample;
%                     end
                end
            else %if Q values zero, choose any random action
                %q_samples=Q1(s,:);
                ak_sample=randperm(length(q_samples),1);
                ak=idx(ak_sample);
            end
            %ak
            [i_idx,j_idx]=find(states==s);
            if ak==1 %update Q value and state based on action taken
                Q1(s,ak)=Q1(s,ak)+alpha_k*((reward(s,ak))+(df*max(Q1(s-1,:))-Q1(s,ak)));
                s=s-1;
            elseif ak==2
                Q1(s,ak)=Q1(s,ak)+alpha_k*((reward(s,ak))+(df*max(Q1(s+10,:))-Q1(s,ak)));
                s=s+10;
            elseif ak==3 
                Q1(s,ak)=Q1(s,ak)+alpha_k*((reward(s,ak))+(df*max(Q1(s+1,:))-Q1(s,ak)));
                s=s+1;
            elseif ak==4
                Q1(s,ak)=Q1(s,ak)+alpha_k*((reward(s,ak))+(df*max(Q1(s-10,:))-Q1(s,ak)));
                s=s-10;

            end
            k=k+1; %update time step
%             if s==100
%                 disp('reached')
%             end
        end
        if Q1==Q
            disp('equal') %if no change in Q matrix values break out of loop
            break;
        end
    end
    
    [~,opt_actions]=max(Q1,[],2); %obtaining optimal policy action values
%%    
    %toc; %ending the timer
    %%
    
%     figure(run)
%     title(['run  ',num2str(run)]);
%     axis([0 10 0 10])
%     grid on , hold on;
%     set(gca,'YDir','reverse')
%     %set(gca,'XDir','reverse')
%     ax=gca;
%     ax.GridColor='k';
%     ax.GridAlpha=0.5;
    
    %text(9.3,0.5,'END','FontSize',14);
    opt_temp={}
    %Calculating total reward
    i=1;
    %opt_temp{end+1}=i;
    z=1;
    total_reward=0;
    while i~=100 && z<=100
%         x=fix(i/10)+0.5;
%         y=10.5-mod(i,10); % locate
        a=opt_actions(i);
        opt_temp{end+1}=i;
        [y,x]=find(states==i);
        if a==1
                %plot(x,y,'^'), hold on;
                total_reward=total_reward+df.^(z-1)*reward(i,1);
                i=i-1;
        elseif a==2
                %plot(x,y,'>'), hold on;
                total_reward=total_reward+df.^(z-1)*reward(i,2);
                i=i+10;
        elseif a==3
                %plot(x,y,'v'), hold on;
                total_reward=total_reward+df.^(z-1)*reward(i,3);
                i=i+1;
        elseif a==4
                %plot(x,y,'<'), hold on;
                total_reward=total_reward+df.^(z-1)*reward(i,4);
                i=i-10;
        end
        z=z+1;
        %i=i+1;
    end % draw path
    i
    
    %%
    if i==100
        
        h=h+1;
    end
    if i==100 && flag==0 && flag1==0
        
        %ave_run_time=[ave_run_time toc];
        %%
        %Generate path of optimal policy
        qevalstates={};
        figure(run)
        title(['run  ',num2str(run), ' Execution of optimal policy with associated reward =',num2str(total_reward)]);
        axis([0 10 0 10])
        grid on , hold on;
        set(gca,'YDir','reverse')
        %set(gca,'XDir','reverse')
        ax=gca;
        ax.GridColor='k';
        ax.GridAlpha=0.5;
        
        %text(9.3,0.5,'END','FontSize',14);
        opt_temp={}
        i=1;
        %opt_temp{end+1}=i;
        z=1;
        %total_reward=0;

        while i~=100 && z<=100
    %         x=fix(i/10)+0.5;
    %         y=10.5-mod(i,10); % locate
            a=opt_actions(i);
            opt_temp{end+1}=i;
            [y,x]=find(states==i);
            qevalstates{end+1}=i;
            if a==1
                    plot(x,y,'^'), hold on;
                    %total_reward=total_reward+gamma.^(z-1)*reward(i,1);
                    i=i-1;
            elseif a==2
                    plot(x,y,'>'), hold on;
                    %total_reward=total_reward+gamma.^(z-1)*reward(i,2);
                    i=i+10;
            elseif a==3
                    plot(x,y,'v'), hold on;
                    %total_reward=total_reward+gamma.^(z-1)*reward(i,3);
                    i=i+1;
            elseif a==4
                    plot(x,y,'<'), hold on;
                    %total_reward=total_reward+gamma.^(z-1)*reward(i,4);
                    i=i-10;
            end
            z=z+1;
            %i=i+1;
        end % draw path
        i
        %%
        %Generating plot of optimal policy
%         figure(run+10)
%         title(['run  ',num2str(run), ' Optimal policy with associated reward =',num2str(total_reward)]);
%         axis([0 10 0 10])
%         grid on , hold on;
%         set(gca,'YDir','reverse')
%         %set(gca,'XDir','reverse')
%         ax=gca;
%         ax.GridColor='k';
%         ax.GridAlpha=0.5;
%         
%         %text(9.3,0.5,'END','FontSize',14);
%         opt_temp={}
%         i=1;
%         %opt_temp{end+1}=i;
%         z=1;
%         total_reward=0;
%         while i~=100 && z<=100
%     %         x=fix(i/10)+0.5;
%     %         y=10.5-mod(i,10); % locate
%             a=opt_actions(i);
%             opt_temp{end+1}=i;
%             [y,x]=find(states==i);
%             if a==1
%                     plot(x,y,'^'), hold on;
%                     %total_reward=total_reward+gamma.^(z-1)*reward(i,1);
%                     %i=i-1;
%             elseif a==2
%                     plot(x,y,'>'), hold on;
%                     %total_reward=total_reward+gamma.^(z-1)*reward(i,2);
%                     %i=i+10;
%             elseif a==3
%                     plot(x,y,'v'), hold on;
%                     %total_reward=total_reward+gamma.^(z-1)*reward(i,3);
%                     %i=i+1;
%             elseif a==4
%                     plot(x,y,'<'), hold on;
%                     %total_reward=total_reward+gamma.^(z-1)*reward(i,4);
%                     %i=i-10;
%             end
%             z=z+1;
%             i=i+1;
%             %i=i+1;
%         end 
        i
        flag=1;
        flag1=1;
    end
    opt{end+1}=opt_temp;
end
qevalstates=cell2mat(qevalstates);
disp("The states obtained in each transition are: ")
disp(qevalstates)