require "nn"
require "cunn"
require 'optim'
require 'cudnn'

-- general parameters
storeCounter = 0 -- counts parameter updates

-- parameters of end to end training
storeInterval = 100 -- storing snapshot after x updates
lrInit = 0.000001 -- learning rate
lrInterval = 25000 --cutting learning rate in half after x updates
gradClamp = 0.001 -- maximum gradient magnitude
oFile = 'obj_model_fcn_e2e.net'

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
    torch.save(oFile, model)
  end
    
  if (storeCounter % lrInterval) == 0 then
    print('TORCH: Cutting learningrate by half. Is now: ' .. optimState.learningRate)
    optimState.learningRate = optimState.learningRate * 0.5
  end

end
