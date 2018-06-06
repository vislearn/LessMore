-- TAKEN FROM https://github.com/davidBelanger/torch-util/blob/master/Entropy.lua

local Entropy, parent = torch.class('nn.Entropy', 'nn.Module')

local eps = 1e-12


--This doesn't assume that each element of input is a single bernoulli probability
--instead, it assumes that each row indexes a distribution. e.g., each row is for a minibatch element. it returns the entropy of each row.

--todo: pass it some flag if you're treating the whole input tensor as a single distribution
function Entropy:__init()
   parent.__init(self)
end

function Entropy:updateOutput(input)
   -- -log(input) * input (and sum over all but the minibatch dimension)
   self.term1 = self.term1 or input.new()
   
   self.term1:resizeAs(input)
  
   self.term1:copy(input):add(eps):log()
   self.term1:cmul(input)

   if(input:dim() == 1) then
       self.output:resize(1)
       self.output[1] = -self.term1:sum()
   else
      local sizePerBatchElement = input:nElement()/input:size(1)
      self.output = self.term1:reshape(input:size(1),sizePerBatchElement):sum(2):mul(-1.0)
   end
   return self.output
end

function Entropy:updateGradInput(input,gradOutput)
   --  d = -(1 + log(x))   
   local d = gradOutput:dim()
   assert(d == 1 or (d == 2 and gradOutput:size(2) == 1))

   self.term2 = self.term2 or input.new()
   self.term2:resizeAs(gradOutput)
   self.term2:copy(gradOutput)

   --the next 4 lines add a bunch of singleton dimensions, which is necessary for the later call to expandAs()
   local s = input:size()
   s[1] = input:size(1)
   for i = 2,s:size() do s[i] = 1 end
   
   self.gradInput:resizeAs(input)
--   self.gradInput:copy(input):add(eps):log():add(1.0):mul(-1.0):cmul(self.term2:reshape(s):expandAs(input))

   self.term2 = torch.expand(self.term2, self.gradInput:size(1))
   self.gradInput:copy(input):add(eps):log():add(1.0):mul(-1.0):cmul(self.term2)

   return self.gradInput
end

