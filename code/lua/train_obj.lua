require 'nn'
require 'cunn'
require 'optim'
require 'cudnn'

-- general parameters
storeCounter = 0 -- counts parameter updates

-- parameters of pretraining
storeInterval = 1000 		-- storing snapshot after x updates
lrInit = 0.0001 		-- initial learning rate
lrInterval = 50000 		-- cutting learning rate in half after x updates
lrIntervalInit = 100000 	-- number if initial iteration without learning rate cutting
gradClamp = 0.5 		-- maximum gradient magnitude (reprojection opt. only)

oFileInit = 'obj_model_fcn_init.net'
oFileRepro = 'obj_model_fcn_repro.net'

mean = {127, 127, 127} 

dofile('MyL1Criterion.lua')

function loadModel(f, inW, inH, outW, outH)

  inputWidth = inW
  inputHeight = inH
  outputWidth = outW
  outputHeight = outH

  print('TORCH: Loading network from file: ' .. f)

  model = torch.load(f)
  model = model:cuda()
  cudnn.convert(model, cudnn)

  model:evaluate()

  criterion = nn.MyL1Criterion()
  criterion = criterion:cuda()

  params, gradParams = model:getParameters()
  optimState = {learningRate = lrInit}
end

function constructModel(inW, inH, outW, outH)

  inputWidth = inW
  inputHeight = inH
  outputWidth = outW
  outputHeight = outH

  print('TORCH: Creating network.')

  -- 640 x 480
  model = nn.Sequential()
  model:add(nn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1)) -- 3
  model:add(nn.ReLU()) 
  model:add(nn.SpatialConvolution(64, 128, 3, 3, 2, 2, 1, 1))  -- 5 
  model:add(nn.ReLU()) 
  -- 320 x 240
  model:add(nn.SpatialConvolution(128, 128, 3, 3, 2, 2, 1, 1)) -- 9
  model:add(nn.ReLU()) 
  -- 160 x 120
  model:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1)) -- 17
  model:add(nn.ReLU()) 
  model:add(nn.SpatialConvolution(256, 256, 3, 3, 2, 2, 1, 1)) -- 19
  model:add(nn.ReLU()) 
  -- 80 x 60
  model:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1)) -- 37
  model:add(nn.ReLU()) 
  model:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)) -- 39
  model:add(nn.ReLU()) 
  model:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)) -- 41
  model:add(nn.ReLU()) 

  model:add(nn.SpatialConvolution(512, 4096, 1, 1, 1, 1, 0, 0))
  model:add(nn.ReLU()) 
  model:add(nn.SpatialConvolution(4096, 4096, 1, 1, 1, 1, 0, 0))
  model:add(nn.ReLU()) 
  model:add(nn.SpatialConvolution(4096, 3, 1, 1, 1, 1, 0, 0))

  criterion = nn.MyL1Criterion()

  model = model:cuda()
  cudnn.convert(model, cudnn)

  model:evaluate()

  criterion = criterion:cuda()

  params, gradParams = model:getParameters()
  optimState = {learningRate = lrInit}
end

function setEvaluate()
    model:evaluate()
    print('TORCH: Set model to evaluation mode.')
end

function setTraining()
    model:training()
    print('TORCH: Set model to training mode.')
end

function forward(count, data)

  local input = torch.FloatTensor(data):reshape(count, 3, inputHeight, inputWidth);
  input = input:cuda()

  -- normalize data
  for c=1,3 do
    input[{ {}, {c}, {}, {}  }]:add(-mean[c]) 
  end

  print('TORCH: Doing a forward pass.')

  local results = model:forward(input)
  results = results:reshape(3, outputHeight * outputWidth):transpose(1,2)
  results = results:double()

  local resultsR = {}
  for i = 1,results:size(1) do
    for j = 1,3 do
      local idx = (i-1) * 3 + j
      resultsR[idx] = results[{i, j}]
    end
  end

  return resultsR
end


function backward(count, loss, data, gradients)

  print('TORCH: Doing a backward pass.')
  local input = torch.FloatTensor(data):reshape(1, 3, inputHeight, inputWidth)
  local dloss_dpred = torch.FloatTensor(gradients):reshape(count, 3):transpose(1,2):reshape(1, 3, outputHeight, outputWidth)

  input = input:cuda()
  dloss_dpred = dloss_dpred:cuda()

  dloss_dpred:clamp(-gradClamp,gradClamp)

  -- normalize data
  for c=1,3 do
    input[{ {}, {c}, {}, {}  }]:add(-mean[c]) 
  end

  gradParams:zero()

  local function feval(params)
    model:backward(input, dloss_dpred)
    return loss,gradParams
  end
  optim.adam(feval, params, optimState)

  storeCounter = storeCounter + 1

  if (storeCounter % storeInterval) == 0 then
    print('TORCH: Storing a snapshot of the network.')
    model:clearState()
    torch.save(oFileRepro, model)
  end

  if storeCounter > (lrIntervalInit - 1) and (storeCounter % lrInterval) == 0 then
    print('TORCH: Cutting learningrate by half. Is now: ' .. optimState.learningRate)
    optimState.learningRate = optimState.learningRate * 0.5

  end
end

function train(data, labels)
  print('TORCH: Doing a training pass.')

  local input = torch.FloatTensor(data):reshape(1, 3, inputHeight, inputWidth)
  local output = torch.FloatTensor(labels):reshape(3, outputHeight * outputWidth):transpose(1,2)
  
  input = input:cuda()
  output = output:cuda()

  -- normalize data
  for c=1,3 do
    input[{ {}, {c}, {}, {}  }]:add(-mean[c]) 
  end

  local loss = 0

  local function feval(params)
    gradParams:zero()

    local pred = model:forward(input)
    pred = pred:reshape(3, outputHeight * outputWidth):transpose(1,2)
    loss = criterion:forward(pred, output)
    local dloss_dpred = criterion:backward(pred, output)
    dloss_dpred = dloss_dpred:transpose(1,2):reshape(1, 3, outputWidth, outputHeight)
    model:backward(input, dloss_dpred)

    return loss,gradParams
  end
  optim.adam(feval, params, optimState)

  storeCounter = storeCounter + 1

  if (storeCounter % storeInterval) == 0 then
    print('TORCH: Storing a snapshot of the network.')
    model:clearState()
    torch.save(oFileInit, model)
  end

  if storeCounter > (lrIntervalInit - 1) and (storeCounter % lrInterval) == 0 then
    print('TORCH: Cutting learningrate by half. Is now: ' .. optimState.learningRate)
    optimState.learningRate = optimState.learningRate * 0.5
  end

  return loss
end
