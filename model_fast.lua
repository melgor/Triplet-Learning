require 'nn'
require 'cunn'
require 'cudnn'
require 'dpnn'

require 'optim'

paths.dofile('torch-TripletEmbedding/TripletEmbedding.lua')

if opt.retrain ~= 'none' then
   paths.dofile(opt.modelDef)
   assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
   print('Loading model from file: ' .. opt.retrain);
   modelAnchor = torch.load(opt.retrain)
else
   paths.dofile(opt.modelDef)
   modelAnchor = createModel(opt.nGPU)
end

criterion = nn.TripletEmbeddingCriterion(opt.alpha)

model = modelAnchor:cuda()
model = makeDataParallel(model, opt.nGPU)
criterion:cuda()

print('=> Model')
print(model)

print('=> Criterion')
print(criterion)

collectgarbage()
