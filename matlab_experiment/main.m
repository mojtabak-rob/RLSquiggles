clear
close all


ObservationInfo = rlNumericSpec([4 1]);
ObservationInfo.Name = 'Rhythmic States';
ObservationInfo.Description = 'b, t, s1, s2';

ActionInfo = rlFiniteSetSpec([0 1]);
ActionInfo.Name = 'Note';

env = rlFunctionEnv(ObservationInfo,ActionInfo,'myStepFunction','myResetFunction');

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

statePath = [
    imageInputLayer([4 1 1], 'Normalization', 'none', 'Name', 'state')
    fullyConnectedLayer(20, 'Name', 'CriticStateFC1')
    reluLayer('Name', 'CriticRelu1')
    fullyConnectedLayer(20, 'Name', 'CriticStateFC2')];
actionPath = [
    imageInputLayer([1 1 1], 'Normalization', 'none', 'Name', 'action')
    fullyConnectedLayer(20, 'Name', 'CriticActionFC1')];
commonPath = [
    additionLayer(2,'Name', 'add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1, 'Name', 'output')];
criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork, actionPath);
criticNetwork = addLayers(criticNetwork, commonPath);    
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');

% set some options for the critic
criticOpts = rlRepresentationOptions('LearnRate',0.01,'GradientThreshold',1);

% create the critic based on the network approximator
critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,...
    'Observation',{'state'},'Action',{'action'},criticOpts);

agentOpts = rlDQNAgentOptions(...
    'UseDoubleDQN',false, ...    
    'TargetUpdateMethod',"periodic", ...
    'TargetUpdateFrequency',4, ...   
    'ExperienceBufferLength',100000, ...
    'DiscountFactor',0.99, ...
    'MiniBatchSize',256);

agent = rlDQNAgent(critic,agentOpts);

trainOpts = rlTrainingOptions;

trainOpts.MaxEpisodes = 100;
trainOpts.MaxStepsPerEpisode = 500;
trainOpts.StopTrainingCriteria = "AverageReward";
trainOpts.StopTrainingValue = 6700;
trainOpts.ScoreAveragingWindowLength = 5;

trainOpts.SaveAgentCriteria = "EpisodeReward";
trainOpts.SaveAgentValue = 500;
trainOpts.SaveAgentDirectory = "savedAgents";

trainOpts.Verbose = false;
trainOpts.Plots = "training-progress";

trainingInfo = train(agent,env,trainOpts);

simOptions = rlSimulationOptions('MaxSteps',500);
experience = sim(env,agent,simOptions);
result1=experience.Action.Note;
result2=experience.Observation.RhythmicStates;

figure
plot(result1)
figure
plot(result2)
figure
plot(experience.Reward)
