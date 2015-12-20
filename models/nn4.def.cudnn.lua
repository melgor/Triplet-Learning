-- Model: nn4.def.lua
-- Description: Implementation of NN4 from the FaceNet paper.
--              Keep 5x5 convolutions because the phrasing
--              "In addition to the reduced input size it
--              does not use 5x5 convolutions in the higher layers"
--              is vague.
-- Input size: 3x96x96
-- Components: Mostly `nn`
-- Devices: CPU and CUDA
--
-- Brandon Amos <http://bamos.github.io>
-- 2015-09-18
--
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
require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
-- need to check CUDNN auto-tuner
-- cudnn.benchmark = true -- run manual auto-tuner provided by cudnn
-- cudnn.verbose = false
local ConvMode = {
                  nil,--'CUDNN_CONVOLUTION_FWD_ALGO_DIRECT',
                  'CUDNN_CONVOLUTION_BWD_DATA_ALGO_0',
                  'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0'
                 }
		 
local SpatialConvolution = cudnn.SpatialConvolution
local SpatialMaxPooling = cudnn.SpatialMaxPooling
local SpatialAveragePooling = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local SpatialBatchNormalization = cudnn.SpatialBatchNormalization

local Inception_cudnn, parent = torch.class("nn.Inception_cudnn", "nn.Decorator")

function Inception_cudnn:__init(config)
   --[[ Required Arguments ]]--
   -- Number of input channels or colors
   self.inputSize = config.inputSize
   -- Number of filters in the non-1x1 convolution kernel sizes, e.g. {32,48}
   self.outputSize = config.outputSize
   -- Number of filters in the 1x1 convolutions (reduction) 
   -- used in each column, e.g. {48,64,32,32}. The last 2 are 
   -- used respectively for the max pooling (projection) column 
   -- (the last column in the paper) and the column that has 
   -- nothing but a 1x1 conv (the first column in the paper).
   -- This table should have two elements more than the outputSize
   self.reduceSize = config.reduceSize
   
   --[[ Optional Arguments ]]--
   -- The strides of the 1x1 (reduction) convolutions. Defaults to {1,1,...}
   self.reduceStride = config.reduceStride or {}
   -- A transfer function like nn.Tanh, nn.Sigmoid, nn.ReLU, nn.Identity, etc. 
   -- It is used after each reduction (1x1 convolution) and convolution
   self.transfer = config.transfer or ReLU(true)   
   -- batch normalization can be awesome
   self.batchNorm = config.batchNorm
   -- Adding padding to the input of the convolutions such that 
   -- input width and height are same as that of output. 
   self.padding = true
   if config.padding ~= nil then
      self.padding = config.padding
   end
   -- The size (height=width) of the non-1x1 convolution kernels. 
   self.kernelSize = config.kernelSize or {5,3}
   -- The stride (height=width) of the convolution. 
   self.kernelStride = config.kernelStride or {1,1}
   -- The size (height=width) of the spatial max pooling used 
   -- in the next-to-last column.
   self.poolSize = config.poolSize or 3
   -- The stride (height=width) of the spatial max pooling.
   self.poolStride = config.poolStride or 1
   -- The pooling layer.
   self.pool = config.pool or SpatialMaxPooling(self.poolSize, self.poolSize, self.poolStride, self.poolStride)
   
   -- [[ Module Construction ]]--
   local depthConcat = nn.DepthConcat(2) -- concat on 'c' dimension
   -- 1x1 conv (reduce) -> 3x3 conv
   -- 1x1 conv (reduce) -> 5x5 conv
   -- ...
   for i=1,#self.kernelSize do
      local mlp = nn.Sequential()
      -- 1x1 conv
      local reduce = SpatialConvolution(
         self.inputSize, self.reduceSize[i], 1, 1, 
         self.reduceStride[i] or 1, self.reduceStride[i] or 1
      ):setMode(unpack(ConvMode)):fastest()
      mlp:add(reduce)
      if self.batchNorm then
         mlp:add(SpatialBatchNormalization(self.reduceSize[i]))
      end
      mlp:add(self.transfer:clone())
      
      -- nxn conv
      local conv = SpatialConvolution(
         self.reduceSize[i], self.outputSize[i], 
         self.kernelSize[i], self.kernelSize[i], 
         self.kernelStride[i], self.kernelStride[i],
         self.padding and math.floor(self.kernelSize[i]/2) or 0, self.padding and math.floor(self.kernelSize[i]/2) or 0
      ):setMode(unpack(ConvMode)):fastest()
      mlp:add(conv)
      if self.batchNorm then
         mlp:add(SpatialBatchNormalization(self.outputSize[i]))
      end
      mlp:add(self.transfer:clone())
      depthConcat:add(mlp)
   end
   
   -- 3x3 max pool -> 1x1 conv
   local mlp = nn.Sequential()
   mlp:add(self.pool)
   -- not sure if transfer should go here? mlp:add(transfer:clone())
   local i = #(self.kernelSize) + 1
   if self.reduceSize[i] then
      local reduce = SpatialConvolution(
         self.inputSize, self.reduceSize[i], 1, 1, 
         self.reduceStride[i] or 1, self.reduceStride[i] or 1
      ):setMode(unpack(ConvMode)):fastest()
      mlp:add(reduce)
      if self.batchNorm then
         mlp:add(SpatialBatchNormalization(self.reduceSize[i]))
      end
      mlp:add(self.transfer:clone())
   end
   depthConcat:add(mlp)
      
   -- reduce: 1x1 conv (channel-wise pooling)
   i = i + 1
   if self.reduceSize[i] then
      local mlp = nn.Sequential()
      local reduce = SpatialConvolution(
          self.inputSize, self.reduceSize[i], 
	         1, 1, 
          self.reduceStride[i] or 1, self.reduceStride[i] or 1
      ):setMode(unpack(ConvMode)):fastest()
      mlp:add(reduce)
      if self.batchNorm then
          mlp:add(SpatialBatchNormalization(self.reduceSize[i]))
      end
      mlp:add(self.transfer:clone())
      depthConcat:add(mlp)
   end
   
   parent.__init(self, depthConcat)
end

function Inception_cudnn:updateOutput(input)
   local input = self:toBatch(input, 3)
   local output = self.module:updateOutput(input)
   self.output = self:fromBatch(output, 3)
   return self.output
end

function Inception_cudnn:updateGradInput(input, gradOutput)
   local input, gradOutput = self:toBatch(input, 3), self:toBatch(gradOutput, 3)
   local gradInput = self.module:updateGradInput(input, gradOutput)
   self.gradInput = self:fromBatch(gradInput, 3)
   return self.gradInput
end

function Inception_cudnn:accGradParameters(input, gradOutput, scale)
   local input, gradOutput = self:toBatch(input, 3), self:toBatch(gradOutput, 3)
   self.module:accGradParameters(input, gradOutput, scale)
end

function Inception_cudnn:accUpdateGradParameters(input, gradOutput, lr)
   local input, gradOutput = self:toBatch(input, 3), self:toBatch(gradOutput, 3)
   self.module:accUpdateGradParameters(input, gradOutput, lr)
end


function createModel(nGPU)
   local net = nn.Sequential()

   net:add(SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3):setMode(unpack(ConvMode)):fastest())
   net:add(SpatialBatchNormalization(64))
   net:add(ReLU(true))

   net:add(SpatialMaxPooling(3, 3, 2, 2, 1, 1))
   -- Don't use normalization.

   -- Inception (2)
   net:add(SpatialConvolution(64, 64, 1, 1, 1, 1, 0, 0):setMode(unpack(ConvMode)):fastest())
   net:add(SpatialBatchNormalization(64))
   net:add(ReLU(true))
   net:add(SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1):setMode(unpack(ConvMode)):fastest())
   net:add(SpatialBatchNormalization(192))
   net:add(ReLU(true))

   -- Don't use normalization.
   net:add(SpatialMaxPooling(3, 3, 2, 2, 1, 1))

   -- Inception (3a)
   net:add(nn.Inception_cudnn{
     inputSize = 192,
     kernelSize = {3, 5},
     kernelStride = {1, 1},
     outputSize = {128, 32},
     reduceSize = {96, 16, 32, 64},
     pool = SpatialMaxPooling(3, 3, 2, 2),
     batchNorm = true
   })

   -- Inception (3b)
   net:add(nn.Inception_cudnn{
     inputSize = 256,
     kernelSize = {3, 5},
     kernelStride = {1, 1},
     outputSize = {128, 64},
     reduceSize = {96, 32, 64, 64},
     pool = nn.SpatialLPPooling(256, 2, 3, 3),
     batchNorm = true
   })

   -- Inception (3c)
   net:add(nn.Inception_cudnn{
     inputSize = 320,
     kernelSize = {3, 5},
     kernelStride = {2, 2},
     outputSize = {256, 64},
     reduceSize = {128, 32, nil, nil},
     pool = SpatialMaxPooling(3, 3, 2, 2):ceil(),
     batchNorm = true
   })
-- 
   -- Inception (4a)
   net:add(nn.Inception_cudnn{
     inputSize = 640,
     kernelSize = {3, 5},
     kernelStride = {1, 1},
     outputSize = {192, 64},
     reduceSize = {96, 32, 128, 256},
     pool = nn.SpatialLPPooling(640, 2, 3, 3),
     batchNorm = true
   })
-- 
   -- Inception (4b)
   net:add(nn.Inception_cudnn{
     inputSize = 640,
     kernelSize = {3, 5},
     kernelStride = {1, 1},
     outputSize = {224, 64},
     reduceSize = {112, 32, 128, 224},
     pool = nn.SpatialLPPooling(640, 2, 3, 3),
     batchNorm = true
   })

   -- Inception (4c)
   net:add(nn.Inception_cudnn{
     inputSize = 640,
     kernelSize = {3, 5},
     kernelStride = {1, 1},
     outputSize = {256, 64},
     reduceSize = {128, 32, 128, 192},
     pool = nn.SpatialLPPooling(640, 2, 3, 3),
     batchNorm = true
   })

   -- Inception (4d)
   net:add(nn.Inception_cudnn{
     inputSize = 640,
     kernelSize = {3, 5},
     kernelStride = {1, 1},
     outputSize = {288, 64},
     reduceSize = {144, 32, 128, 160},
     pool = nn.SpatialLPPooling(640, 2, 3, 3),
     batchNorm = true
   })

   -- Inception (4e)
   net:add(nn.Inception_cudnn{
     inputSize = 640,
     kernelSize = {3, 5},
     kernelStride = {2, 2},
     outputSize = {256, 128},
     reduceSize = {160, 64, nil, nil},
     pool = SpatialMaxPooling(3, 3, 2, 2):ceil(),
     batchNorm = true
   })

   -- Inception (5a)
   net:add(nn.Inception_cudnn{
     inputSize = 1024,
     kernelSize = {3, 5},
     kernelStride = {1, 1},
     outputSize = {384, 128},
     reduceSize = {192, 48, 128, 384},
     pool = nn.SpatialLPPooling(960, 2, 3, 3),
     batchNorm = true
   })

   -- Inception (5b)
   net:add(nn.Inception_cudnn{
     inputSize = 1024,
     kernelSize = {3, 5},
     kernelStride = {1, 1},
     outputSize = {384, 128},
     reduceSize = {192, 48, 128, 384},
     pool = SpatialMaxPooling(3, 3, 2, 2),
     batchNorm = true
   })

   net:add(SpatialAveragePooling(3, 3))

   -- Validate shape with:
   -- net:add(nn.Reshape(1024))

   net:add(nn.View(1024))
   net:add(nn.Linear(1024, 128))
   net:add(nn.Normalize(2))
--    
--    print(#net:cuda():forward(torch.CudaTensor(1,3,96,96)))

   return net
end
