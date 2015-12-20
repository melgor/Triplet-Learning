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

   outDim = 128
   local numImages = inputsCPU:size(1)
   local embeddings = torch.Tensor(numImages, outDim)
   local beginIdx = 1
   local inputs = torch.CudaTensor()
   while beginIdx <= numImages do
      local endIdx = math.min(beginIdx+opt.batchSize-1, numImages)
      local range = {{beginIdx,endIdx}}
      local sz = inputsCPU[range]:size()
      inputs:resize(sz):copy(inputsCPU[range])
      local reps = model:forward(inputs):float()
      embeddings[range] = reps

      beginIdx = endIdx + 1
   end
   assert(beginIdx - 1 == numImages)

      
   --calcualate the number of all possible pairs
   local allPossiblePairAnchorPositive = 0
   for i = 1,opt.peoplePerBatch do   
    local n = numPerClass[i]
    for j = 1,n-1 do
      for pair = j,n-1 do
        allPossiblePairAnchorPositive = allPossiblePairAnchorPositive + 1
      end
    end
   end
   -- print("All possible triplet: " .. allPossiblePairAnchorPositive)

   -- local numTrips = numImages - opt.peoplePerBatch
   local numTrips  = allPossiblePairAnchorPositive 
   -- local as = torch.Tensor(numTrips, outDim)
   -- local ps = torch.Tensor(numTrips, outDim)
   -- local ns = torch.Tensor(numTrips, outDim)

   local as_table = {}
   local ps_table = {}
   local ns_table = {}

   local triplet_idx = {}
   local num_example_per_idx = torch.Tensor(embeddings:size(1))
   num_example_per_idx:zero()

   local alpha = 0.2
   local tripIdx = 1
   local shuffle = torch.randperm(numTrips)
   local embStartIdx = 1
   local randomNegNum = 0
   for i = 1,opt.peoplePerBatch do
      local n = numPerClass[i]
      for j = 1,n-1 do --for every image in batch
        local aIdx = embStartIdx + j -1
        local diff = embeddings - embeddings[{ {aIdx} }]:expandAs(embeddings)
        local norms = diff:norm(2, 2):squeeze()    
        for pair = j,n-1 do --create all posible positive pairs
          local pIdx = embStartIdx + pair
          -- Select a semi-hard negative that has a distance
          -- further away from the positive exemplar.

          local selNegIdx = embStartIdx
          while selNegIdx >= embStartIdx and selNegIdx <= embStartIdx+n-1 do
              selNegIdx = (torch.random() % numImages) + 1
          end
          local randomNeg = true
          
          --choose random example which is in margin 
          local normsP = norms - torch.Tensor(embeddings:size(1)):fill((embeddings[aIdx]-embeddings[pIdx]):norm())
          --clean the idx of same class
          normsP[{{embStartIdx,embStartIdx +n-1}}] = normsP:max()
          -- get indexes of example which are inside margin
          local in_margin = normsP:lt(alpha)
	  local allNeg = torch.find(in_margin, 1)
          if table.getn(allNeg) ~= 0 then 
              selNegIdx = allNeg[math.random (table.getn(allNeg))] 
              randomNeg = false
          end

          --use only non-random triplets. Random triples (which are beyond margin) will just produce gradient = 0, so averege gradient will decrease
          if randomNeg == true then 
            randomNegNum = randomNegNum + 1 
          else
            --get embeding of each example 
            -- as[tripIdx] = embeddings[aIdx]
            -- ps[tripIdx] = embeddings[pIdx]
            -- ns[tripIdx] = embeddings[selNegIdx]
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
        end
      end
      embStartIdx = embStartIdx + n
   end
   assert(embStartIdx - 1 == numImages)
   -- assert(tripIdx - 1 == numTrips)
   print(('  + (nRandomNegs, nTrips, nTripsRight) = (%d, %d, %d)'):format(randomNegNum, numTrips,table.getn(as_table)))
   local as = torch.concat(as_table):view(table.getn(as_table),outDim)
   local ps = torch.concat(ps_table):view(table.getn(ps_table),outDim)
   local ns = torch.concat(ns_table):view(table.getn(ns_table),outDim)

   local beginIdx = 1
   local inCuda = torch.CudaTensor()
   local asCuda = torch.CudaTensor()
   local psCuda = torch.CudaTensor()
   local nsCuda = torch.CudaTensor()

   -- Return early if the loss is 0 for `numZeros` iterations.
   local numZeros = 4
   local zeroCounts = torch.IntTensor(numZeros):zero()
   local zeroIdx = 1

   -- Return early if the loss shrinks too much.
   -- local firstLoss = nil

   -- TODO: Should be <=, but batches with just one image cause errors.
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

    -- Return early if the loss is 0 for `numZeros` iterations.
    zeroCounts[zeroIdx] = (err == 0.0) and 1 or 0 -- Boolean to int.
    zeroIdx = (zeroIdx % numZeros) + 1
    if zeroCounts:sum() == numZeros then
       return
    end
end
