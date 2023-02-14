%%
%Loading the reward function and defining the maximum number of trials,runs
%and time steps
close all
load('C:\Users\Varshini\Downloads\RL_ProjectDescription\RL_Project\task1.mat')
Q1=zeros(size(reward));
trial=1;
e=1e-4; 
k=1;
s1=1;
df=0.9;
%ek=100/(100+k);
%alpha_k=ek;
runs=10; %defining maximum number of runs
trial_max=3000; %defining maximum number of trials
k_max=100; %defining maximum number of time steps.
goals={};
reached_goals=0;
states=zeros(10,10); %saving the states to a 10*10 matrix that represents the 10*10 grid.
k=1;
for j=1:size(states,2)
    for i=1:size(states,2)
        states(i,j)=k;
        k=k+1;
    end
end
%goals=zeros(runs,1);
opt={};
%%
%Implementing Q Learning Algorithm
prob=[1/k; 100/(100+k); (1+log(k))/k;(1+5*log(k))/k;]; %defining the probability functions
discount_factors=[0.5;0.9]; %defining the discount factors
results={}; %defining the cell array that stores the results of these runs
fig=1;
for m=1:length(prob) %iterating through each probability function
    
    
    for n=1:length(discount_factors) %iterating through each discount factor
        
        df=discount_factors(n);
        total_time=0;
        h=0;
        flag=0;
        flag1=0;
        for run=1:runs
            run
            
            tic;
            Q=zeros(size(reward));
            Q1=zeros(size(reward)); %initializing Q1 and Q matrix that stores the Q values of the state in each time step
            
            for trial=1:trial_max
                Q=Q1; %setting Q to Q1 for comparisom
                %trial
                k=1; 
                s=1;
%                 ek=(100)/(100+k);
%                 alpha_k=ek;
                while s~=100
        %             if s==100
        %                 goals(run)=1;
        %                 %fprintf("breaking")
        %                 break
        %             end
        
                    %ek=(1+(5*log(k)))/(k);
                    %ek=prob(m)
                    if m==1 %obtaining the probabilty function
                        ek=1/k;
                    elseif m==2
                        ek=100/(100+k);
                    elseif m==3
                        ek=(1+log(k))/k;
                    elseif m==4
                        ek=(1+(5*log(k)))/k;
                    end
                      
                    alpha_k=ek; %setting learning rate equal to the probability factor
                    if alpha_k<0.005 %setting condition to break from loop if learning rates reaches 0.005
                        %disp('breaking')
                        break;
                    end
                    [i_idx,j_idx]=find(states==s); %find indices that correspond to present state
                    idx_vals=ones(length(Q1(s,:)),1);
                    if i_idx==1 %condition to check if state is located in the first row of 10*10 grid
                        idx_vals(1)=0; %set corresponding action index point to 0 if the given state is a boundary point
                    end
                    if j_idx==size(states,2) %condition to check if state is located in the last column of 10*10 grid
                        idx_vals(2)=0; %set corresponding action index point to 0 if the given state is a boundary point
                    end
                    if i_idx==size(states,1)%condition to check if state is located in the last row of 10*10 grid
                        idx_vals(3)=0; %set corresponding action index point to 0 if the given state is a boundary point
                    end
                    if j_idx==1 %condition to check if state is located in the first column of 10*10 grid
                        idx_vals(4)=0; %set corresponding action index point to 0 if the given state is a boundary point
                    end
                    p=rand; %generating random unique value from 0 to 1
                    idx=find(idx_vals==1);
                    %idx=find(reward(s,:)~=-1);
                    q_samples=Q1(s,idx);
                    if any(Q1(s,:)) %checking if Q values are zero or not; exploration/exploitation performed based on probability if non zero elements present
                        
                        if p>=ek %if probability value is lesser than probability factor value at current time step choose action with maximum Q value
                            
                            idx_max=find(q_samples==max(q_samples));
                            ak_max_sample=randperm(length(idx_max),1); %if more than one action having same maximum Q value, choose index randomly
                            ak=idx(idx_max(ak_max_sample));
                        else
                            %q_samples=Q1(s,:);
                            idx_rand=find(q_samples~=max(q_samples)); %choose random action not having maximum Q value if probability is greater.
                            %q_rand=q_samples(idx_rand);
                            %if length(idx_rand)~=0
        
                            ak_sample=randperm(length(idx_rand),1);
                            ak=idx(idx_rand(ak_sample));
        %                     else
        %                         ak_sample=randperm(length(q_samples),1);
        %                         ak=ak_sample;
        %                     end
                        end
                    else %if Q1 matrix empty, choose any random action
                        %q_samples=Q1(s,:);
                        ak_sample=randperm(length(q_samples),1);
                        ak=idx(ak_sample);
                    end
                    %ak
                    [i_idx,j_idx]=find(states==s);
                    if ak==1 %choosing next state based on action
                        Q1(s,ak)=Q1(s,ak)+alpha_k*((reward(s,ak))+(df*max(Q1(s-1,:))-Q1(s,ak))); %updating Q value of state
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
                    k=k+1; %incrementing time step
        %             if s==100
        %                 disp('reached')
        %             end
                end
                if Q1==Q
                    disp('equal') %if equal, convergence is indicated and thus loop will get broken if satisfied.
                    break;
                end
            end
            %%
            %obtainiing optimal policy and total rewards using the Q values
            %obtained
            [~,opt_actions]=max(Q1,[],2); %obtaining array containing optimal actions at each state.
            
            toc; %ending the timer
            i=1; 
            %opt_temp{end+1}=i;
            z=1;
            total_reward=0; %initializing sum of total reward to 0.
            opt_temp={};
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
            end 
            total_time=total_time+toc; %obtaining total time

            if i==100 && flag==0 && flag1==0
                h=h+1; %updating value if end state is reached 
                %%
                %Plotting Optimal Policy Path
                figure(run)
                %fig=fig+1;
                title(['run  ',num2str(run), ' Execution of optimal policy with associated reward =',num2str(total_reward)]);
                axis([0 10 0 10])
                grid on , hold on;
                set(gca,'YDir','reverse')
                %set(gca,'XDir','reverse')
                ax=gca;
                ax.GridColor='k';
                ax.GridAlpha=0.5;
                
                %text(9.3,0.5,'END','FontSize',14);
                
                i=1;
                %opt_temp{end+1}=i;
                z=1;
                %total_reward=0;
                while i~=100 && z<=100
            %         x=fix(i/10)+0.5;
            %         y=10.5-mod(i,10); % locate
                    a=opt_actions(i);
                    %opt_temp{end+1}=i;
                    [y,x]=find(states==i);
                    if a==1
                            plot(x,y,'^'), hold on;
                            %total_reward=total_reward+df.^(z-1)*reward(i,1);
                            i=i-1;
                    elseif a==2
                            plot(x,y,'>'), hold on;
                            %total_reward=total_reward+df.^(z-1)*reward(i,2);
                            i=i+10;
                    elseif a==3
                            plot(x,y,'v'), hold on;
                            %total_reward=total_reward+df.^(z-1)*reward(i,3);
                            i=i+1;
                    elseif a==4
                            plot(x,y,'<'), hold on;
                            %total_reward=total_reward+df.^(z-1)*reward(i,4);
                            i=i-10;
                    end
                    z=z+1;
                    %i=i+1;
                end % draw path
                i
                %%
                %Plotting Optimal Policy of Each State
                figure(fig+10)
                fig=fig+1;
                title(['run  ',num2str(run), ' Optimal policy with associated reward =',num2str(total_reward)]);
                axis([0 10 0 10])
                grid on , hold on;
                set(gca,'YDir','reverse')
                %set(gca,'XDir','reverse')
                ax=gca;
                ax.GridColor='k';
                ax.GridAlpha=0.5;
                
                %text(9.3,0.5,'END','FontSize',14);
                
                i=1;
                %opt_temp{end+1}=i;
                z=1;
                %total_reward=0;
                while i~=100 && z<=100
            %         x=fix(i/10)+0.5;
            %         y=10.5-mod(i,10); % locate
                    a=opt_actions(i);
                    %opt_temp{end+1}=i;
                    [y,x]=find(states==i);
                    if a==1
                            plot(x,y,'^'), hold on;
                            %total_reward=total_reward+df.^(z-1)*reward(i,1);
                            %i=i-1;
                    elseif a==2
                            plot(x,y,'>'), hold on;
                            %total_reward=total_reward+df.^(z-1)*reward(i,2);
                            %i=i+10;
                    elseif a==3
                            plot(x,y,'v'), hold on;
                            %total_reward=total_reward+df.^(z-1)*reward(i,3);
                            %i=i+1;
                    elseif a==4
                            plot(x,y,'<'), hold on;
                            %total_reward=total_reward+df.^(z-1)*reward(i,4);
                            %i=i-10;
                    end
                    z=z+1;
                    i=i+1;
                    %i=i+1;
                end % draw path
                i
                    %ave_run_time=[ave_run_time toc];
                flag=1;
                flag1=1;
            end
            
        end
        results_temp={}
        results_temp{end+1}=m;
        results_temp{end+1}=df;
        results_temp{end+1}=h;
        results_temp{end+1}=total_time/runs;
        results{end+1}=results_temp;
        %results_temp{end+1}=total_reward;
        %opt{end+1}=opt_temp;
    end
end
