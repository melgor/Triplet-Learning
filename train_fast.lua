-- Copyright 2015 Carnegie Mellon University
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

require 'optim'
require 'fbnn'
require 'image'
require 'torchx' --for concetration the table of tensors

paths.dofile("OpenFaceOptim.lua")


local optimMethod = optim.adadelta
local optimState = {} -- Use for other algorithms like SGD
local optimator = OpenFaceOptim(model, optimState)

trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))

local batchNumber
local triplet_loss

function train()
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)

   batchNumber = 0
   cutorch.synchronize()

   -- set the dropouts to training mode
   model:training()
   if opt.nGPU == 1 then
    model:cuda() -- get it back on the right GPUs.
   end

   local tm = torch.Timer()
   triplet_loss = 0

   local i = 1
   while batchNumber < opt.epochSize do
      -- queue jobs to data-workers
      donkeys:addjob(
         -- the job callback (runs in data-worker thread)
         function()
            local inputs, numPerClass = trainLoader:samplePeople(opt.peoplePerBatch,
                                                                 opt.imagesPerPerson)
            inputs = inputs:float()
            numPerClass = numPerClass:float()
            return sendTensor(inputs), sendTensor(numPerClass)
         end,
         -- the end callback (runs in the main thread)
         trainBatch
      )
      if i % 5 == 0 then
         donkeys:synchronize()
      end
      i = i + 1
   end

   donkeys:synchronize()
   cutorch.synchronize()

   triplet_loss = triplet_loss / batchNumber

   trainLogger:add{
      ['avg triplet loss (train set)'] = triplet_loss,
   }
   print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                          .. 'average triplet loss (per batch): %.2f',
                       epoch, tm:time().real, triplet_loss))
   print('\n')

   collectgarbage()

   local function sanitize(net)
      net:apply(function (val)
            for name,field in pairs(val) do
               if torch.type(field) == 'cdata' then val[name] = nil end
               if name == 'homeGradBuffers' then val[name] = nil end
               if name == 'input_gpu' then val['input_gpu'] = {} end
               if name == 'gradOutput_gpu' then val['gradOutput_gpu'] = {} end
               if name == 'gradInput_gpu' then val['gradInput_gpu'] = {} end
               if (name == 'output' or name == 'gradInput')
               and torch.type(field) == 'torch.CudaTensor' then
                  cutorch.withDevice(field:getDevice(), function() val[name] = field.new() end)
               end
            end
      end)
   end
   sanitize(model)
   if opt.nGPU == 1 then
     torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'),model)
   else
    torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'),model.modules[1])
   end
   torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
   collectgarbage()
end -- of train()

local inputsCPU = torch.FloatTensor()
local numPerClass = torch.FloatTensor()

local timer = torch.Timer()
function trainBatch(inputsThread, numPerClassThread)
   if batchNumber >= opt.epochSize then
    return
  end

  cutorch.synchronize()
  timer:reset()
  receiveTensor(inputsThread, inputsCPU)
  receiveTensor(numPerClassThread, numPerClass)

  local numImages = inputsCPU:size(1)
  local embeddings = model:forward(inputsCPU:cuda()):float()

  local as_table = {}
  local ps_table = {}
  local ns_table = {}

  local triplet_idx = {}
  local num_example_per_idx = torch.Tensor(embeddings:size(1))
  num_example_per_idx:zero()

  local tripIdx = 1
  local embStartIdx = 1
  local numTrips = 0
  for i = 1,opt.peoplePerBatch do
    local n = numPerClass[i]
    for j = 1,n-1 do --for every image in batch
      local aIdx = embStartIdx + j - 1
      local diff = embeddings - embeddings[{ {aIdx} }]:expandAs(embeddings)
      local norms = diff:norm(2, 2):pow(2):squeeze()    --L2 norm have be squared
      for pair = j,n-1 do --create all possible positive pairs
        local pIdx = embStartIdx + pair
        -- Select a semi-hard negative that has a distance
        -- further away from the positive exemplar. Oxford-Face Idea

        --choose random example which is in margin
        local fff = (embeddings[aIdx]-embeddings[pIdx]):norm(2)
        local normsP = norms - torch.Tensor(embeddings:size(1)):fill(fff*fff)  --L2 norm should be squared
        --clean the idx of same class by setting to them max value
        normsP[{{embStartIdx,embStartIdx +n-1}}] = normsP:max()
        -- get indexes of example which are inside margin
        local in_margin = normsP:lt(opt.alpha)
        local allNeg = torch.find(in_margin, 1)

        if table.getn(allNeg) ~= 0 then  --use only non-random triplets. Random triples (which are beyond margin) will just produce gradient = 0, so average gradient will decrease
          selNegIdx = allNeg[math.random (table.getn(allNeg))]
          --get embeding of each example
          table.insert(as_table,embeddings[aIdx])
          table.insert(ps_table,embeddings[pIdx])
          table.insert(ns_table,embeddings[selNegIdx])
          -- get original idx of triplets
          table.insert(triplet_idx,{aIdx,pIdx,selNegIdx})
          -- increase number of times of using each example, need for averaging then
          num_example_per_idx[aIdx] = num_example_per_idx[aIdx] + 1
          num_example_per_idx[pIdx] = num_example_per_idx[pIdx] + 1
          num_example_per_idx[selNegIdx] = num_example_per_idx[selNegIdx] + 1
          tripIdx = tripIdx + 1
        end

        numTrips = numTrips + 1
      end
    end
    embStartIdx = embStartIdx + n
  end
  assert(embStartIdx - 1 == numImages)
  print(('  + (nTrips, nTripsRight) = (%d, %d)'):format(numTrips,table.getn(as_table)))

  local as = torch.concat(as_table):view(table.getn(as_table),opt.embSize)
  local ps = torch.concat(ps_table):view(table.getn(ps_table),opt.embSize)
  local ns = torch.concat(ns_table):view(table.getn(ns_table),opt.embSize)

  local beginIdx = 1
  local inCuda = torch.CudaTensor()
  local asCuda = torch.CudaTensor()
  local psCuda = torch.CudaTensor()
  local nsCuda = torch.CudaTensor()

  local sz = as:size()
  inCuda = inputsCPU:cuda()
  asCuda:resize(sz):copy(as)
  psCuda:resize(sz):copy(ps)
  nsCuda:resize(sz):copy(ns)
  local err, outputs = optimator:optimizeTripletFast(optimMethod,
                                                 inCuda,
                                                 {asCuda, psCuda, nsCuda},
                                                 criterion, triplet_idx, num_example_per_idx)

  -- DataParallelTable's syncParameters
  model:apply(function(m) if m.syncParameters then m:syncParameters() end end)
  cutorch.synchronize()
  batchNumber = batchNumber + 1
  print(('Epoch: [%d][%d/%d]\tTime %.3f\ttripErr %.2e'):format(
        epoch, batchNumber, opt.epochSize, timer:time().real, err))
  timer:reset()
  triplet_loss = triplet_loss + err
end
