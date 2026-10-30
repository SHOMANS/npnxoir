% Close all figure windows before starting
close all

% Backpropagtion learning.  One hidden layer.
% by Dave Touretzky (modified by Nikolay Nikolaev)
% https://www.cs.cmu.edu/afs/cs/academic/class/15782-f06/matlab/
% Uses the following global variables for input and/or output:
%    Inputs1   -  input patterns
%    Desired   -  desired output patterns
%    LearnRate -  learning rate parameter
%    Weights1  -  first weight layer (updated by this routine)
%    Weights2  -  second weight layer (updated by this routine)
%    deltaW1   -  initialize to 0 before first call
%    deltaW2   -  initialize to 0 before first call
load XorPats.dat
load XorAns.dat
Patterns = XorPats';
Desired = XorAns';
[NINPUTS,NPATS] = size(Patterns);
NHIDDENS = 2;
[NOUTPUTS,NPATS] = size(Desired);
PlotBpPats(Patterns,Desired);
LearnRate = 0.25; Momentum = 0; DerivIncr = 0;
deltaW1 = 0; deltaW2 = 0;
Inputs1 = [ones(1,NPATS); Patterns];
Weights1 = rand(NHIDDENS,1+NINPUTS)-0.5;
Weights2 = rand(NOUTPUTS,1+NHIDDENS)-0.5;
TSS_Limit = 0.02
for epoch = 1:1000
  % Forward propagation
  NetIn1 = Weights1 * Inputs1;
  Hidden = 1.0 ./( 1.0 + exp( -NetIn1 ));
  Inputs2 = [ones(1,NPATS); Hidden];
  NetIn2 = Weights2 * Inputs2;
  Out = 1.0 ./ ( 1.0 + exp( -NetIn2 ));
  % Backward propagation of errors
  Error = Desired - Out;
  TSS = sum(sum( Error.^2 ));
  Beta = Out .* (1.0 - Out) .* Error;
  bperr = ( Weights2' * Beta );
  HiddenBeta = Hidden .* (1.0 - Hidden) .* bperr(2:end,:);
  % Calculate the weight updates:
  dW2 = Beta * Inputs2';
  dW1 = HiddenBeta * Inputs1';
  deltaW2 = LearnRate * dW2 + Momentum * deltaW2;
  deltaW1 = LearnRate * dW1 + Momentum * deltaW1;
  % Update the weights:
  Weights2 = Weights2 + deltaW2;
  Weights1 = Weights1 + deltaW1;
  fprintf('Epoch %3d:  Error = %f\n',epoch,TSS);
  if TSS < TSS_Limit, break, end
  if rem(epoch-1,20)==0, PlotBpBoundary(Weights1,epoch,'--'), end
end
PlotBpBoundary(Weights1,epoch,'-')
