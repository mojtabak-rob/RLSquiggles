function [InitialObservation,LoggedSignals] = myResetFunction()
% Reset function to place custom cart-pole environment into a random
% initial state.

b=0;
t=0;

s1=0;
s2=0;

% Return initial environment state variables as logged signals.
LoggedSignals.State = [b;t;s1;s2];
InitialObservation = LoggedSignals.State;

end
