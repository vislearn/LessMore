require "nn"
require "cunn"
require 'optim'
require 'cudnn'

-- general parameters
storeCounter = 0
lrInitE2E = 0.001

dofile('Entropy.lua')

function loadModel(t, outW, outH)

  outputWidth = outW
  outputHeight = outH

  print('TORCH: Loading Score.')

  -- position and softness of inlier threshold
  inlierThresh = t
  inlierSoft = 0.5 -- sigmoid softness (beta)
  etarget = 6 -- target entropy

  print('TORCH: Inlier threshold: ' .. inlierThresh)
  print('TORCH: Target entropy: ' .. etarget)

  model = nn.Sequential()

  -- apply inlier threshold (non-learnable)
  model:add(nn.AddConstant(-inlierThresh))
  model:add(nn.MulConstant(inlierSoft))

  model:add(nn.Sigmoid())

  -- inliers is 1 - sigmoid()
  model:add(nn.MulConstant(-1))
  model:add(nn.AddConstant(1))

  model:add(nn.Sum(3)) 
  model:add(nn.Sum(3)) 

  model:add(nn.Mul())

  model = model:cuda()
  cudnn.convert(model, cudnn)

  model:evaluate()

  params, gradParams = model:getParameters()
  optimState = {learningRate = lrInitE2E}

  params[{1}] = 0.1
  gradParams:zero()

  criterion = nn.AbsCriterion()
  criterion = criterion:cuda()

  -- stuff for entropy controlled training
  emodel = nn.Sequential()
  emodel:add(nn.SoftMax())
  emodel:add(nn.Entropy())
  emodel:add(nn.MulConstant(1/math.log(2)))

  emodel = emodel:cuda()
  cudnn.convert(emodel, cudnn)

  ecriterion = nn.MSECriterion()
  ecriterion = ecriterion:cuda()

  model2 = model:clone()
  params2, gradParams2 = model2:getParameters()
end

function setEvaluate()
    model:evaluate()
    print('TORCH: Set score to evaluation mode.')
end

function setTraining()
    model:training()
    print('TORCH: Set score to training mode.')
end

function forward(count, data)
  print('TORCH: Doing a forward pass for ' .. count .. ' images.')
  local input = torch.FloatTensor(data):reshape(count, 1, outputHeight, outputWidth);
  input = input:cuda()

  local results = model:forward(input)

  local r = {}
  for i = 1,results:size(1) do
    if count == 1 then
      r[i] = results[{i}]
    else
      r[i] = results[{i, 1}]
    end
  end

  return unpack(r)
end

function backward(count, data, outputGradients)
  print('TORCH: Doing a backward pass for ' .. count .. ' images.')

  local input = torch.FloatTensor(data):reshape(count, 1, outputHeight, outputWidth);
  input = input:cuda()

  local gradOutput = torch.FloatTensor(outputGradients):reshape(count, 1);
  gradOutput = gradOutput:cuda()

  local gradInput = model:backward(input, gradOutput)
  gradInput = gradInput:double()
 
  storeCounter = storeCounter + 1

  -- entropy control
  local scores = model2:forward(input)
  scores = scores:reshape(scores:size()[1])

  local eresult = emodel:forward(scores)

  local egt = eresult:clone()
  egt[1] = etarget

  local eloss = ecriterion:forward(eresult, egt)
  local ecritgrad = ecriterion:backward(eresult, egt) -- gradient of the loss
  local emodelgrad = emodel:backward(scores, ecritgrad) -- gradient of the entropy

  emodelgrad = emodelgrad:reshape(1, scores:size()[1])
  model2:backward(input, emodelgrad)

  -- insert optimizer here
  local function feval(params2)
    return 0, gradParams2
  end

  gradParams2:mul(0.1)

  optim.adam(feval, params2, optimState)

  print('Current score scale:')
  print(params)

  params:copy(params2)

  gradParams2:zero()
  gradParams:zero()

  local gradInputR = {}
  for c = 1,count do
    for x = 1,outputWidth do
      for y = 1,outputHeight do
         local idx = (c-1) * outputHeight * outputWidth + (x-1) * outputHeight + y
         gradInputR[idx] = gradInput[{c, 1, y, x}]
      end
    end
  end

  return gradInputR
end
