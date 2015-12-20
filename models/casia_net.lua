-- Model: nn4.def.lua
-- Description: Implementation of CASIA-Net from the "Learning Face Representation from Scratch" paper.
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

local ConvMode = {
                  nil,--'CUDNN_CONVOLUTION_FWD_ALGO_DIRECT',
                  'CUDNN_CONVOLUTION_BWD_DATA_ALGO_0',
                  'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0'
                 }
		 
local SpatialConvolution = cudnn.SpatialConvolution
local SpatialMaxPooling = cudnn.SpatialMaxPooling
local SpatialAveragePooling = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU



function createModel(nGPU)
   local net = nn.Sequential()
   --stage 1
   net:add(SpatialConvolution(3, 32, 3, 3, 1, 1, 1, 1):setMode(unpack(ConvMode)):fastest())
   net:add(nn.SpatialBatchNormalization(32))
   net:add(ReLU(true))
   net:add(SpatialConvolution(32, 64, 3, 3, 1, 1, 1, 1):setMode(unpack(ConvMode)):fastest())
   net:add(nn.SpatialBatchNormalization(64))
   net:add(ReLU(true))
   net:add(SpatialMaxPooling(2, 2, 2, 2, 1, 1))
   
   --stage 2
   net:add(SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1):setMode(unpack(ConvMode)):fastest())
   net:add(nn.SpatialBatchNormalization(64))
   net:add(ReLU(true))
   net:add(SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1):setMode(unpack(ConvMode)):fastest())
   net:add(nn.SpatialBatchNormalization(128))
   net:add(ReLU(true))
   net:add(SpatialMaxPooling(2, 2, 2, 2, 1, 1))
   
   --stage 3
   net:add(SpatialConvolution(128, 96, 3, 3, 1, 1, 1, 1):setMode(unpack(ConvMode)):fastest())
   net:add(nn.SpatialBatchNormalization(96))
   net:add(ReLU(true))
   net:add(SpatialConvolution(96, 192, 3, 3, 1, 1, 1, 1):setMode(unpack(ConvMode)):fastest())
   net:add(nn.SpatialBatchNormalization(192))
   net:add(ReLU(true))
   net:add(SpatialMaxPooling(2, 2, 2, 2, 1, 1))
   
   --stage 4
   net:add(SpatialConvolution(192, 128, 3, 3, 1, 1, 1, 1):setMode(unpack(ConvMode)):fastest())
   net:add(nn.SpatialBatchNormalization(128))
   net:add(ReLU(true))
   net:add(SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1):setMode(unpack(ConvMode)):fastest())
   net:add(nn.SpatialBatchNormalization(256))
   net:add(ReLU(true))
   net:add(SpatialMaxPooling(2, 2, 2, 2, 1, 1))
   
   --stage 5
   net:add(SpatialConvolution(256, 160, 3, 3, 1, 1, 1, 1):setMode(unpack(ConvMode)):fastest())
   net:add(nn.SpatialBatchNormalization(160))
   net:add(ReLU(true))
   net:add(SpatialConvolution(160, 320, 3, 3, 1, 1, 1, 1):setMode(unpack(ConvMode)):fastest())
   net:add(nn.SpatialBatchNormalization(320))
   net:add(ReLU(true))
   net:add(SpatialMaxPooling(2, 2, 2, 2, 1, 1))

   net:add(SpatialAveragePooling(4, 4))

   -- Validate shape with:
   -- net:add(nn.Reshape(320))

   net:add(nn.View(320))
   net:add(nn.Linear(320, 128))
   net:add(nn.Normalize(2))
  
   
   -- print(#net:cuda():forward(torch.CudaTensor(1,3,96,96)))
   
   
   return net
end
