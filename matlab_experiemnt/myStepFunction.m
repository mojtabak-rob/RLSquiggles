function [NextObs,Reward,IsDone,LoggedSignals] = myStepFunction(Action,LoggedSignal)
% Custom step function to construct cart-pole environment for the function
% name case.
%
% This function applies the given action to the environment and evaluates
% the system dynamics for one simulation step.
time_size=490;

% beats_time=[100 175 200 275 300 375 400 475];
beats_time=[70 90 110 140 160 180 210 230 250 280 300 320 350 370 390 420 440 460];

[a,beats_num]=size(beats_time);


% Define the environment constants.
State=LoggedSignal.State;

b_1=State(1);
t_1=State(2);
s1_1=State(3);
s2_1=State(4);



t=t_1+1;

b=0;
for i=1:beats_num
    if t==beats_time(i)
        b=1;
    end
end


if b==1
    s1=0;
    s2=s1_1+1;
    if s1_1==0
        s2=0;
    end
else
    s1=s1_1+1;
    if s2_1>0
        s2=s2_1+1;
    else
        s2=0;
    end
end

if s1_1==0 && b_1==0
    s1=0;
    s2=0;
end





if Action==0
    Reward=-1;
else
    if b_1==1
        Reward=400;
    else
        Reward=-s1_1;
        if s1_1==0
            Reward=-30;
        end
    end
end


% if b==1
%     if Action==b
%         Reward=100;
%     else
%         Reward=-100;
%     end
% else
%     if Action==b
%         Reward=1;
%     else
%         Reward=-s1_1;
%         if s1_1==0
%             Reward=-30;
%         end
%     end
% end


% Perform Euler integration.
LoggedSignals.State = [b;t;s1;s2];

% Transform state to observation.
NextObs = LoggedSignals.State;

% Check terminal condition.

IsDone = t>time_size;

% Get reward.


end