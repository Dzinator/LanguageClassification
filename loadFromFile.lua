require 'nn';
require 'image';
require 'optim';
require 'gnuplot';

model = require 'model.lua'
print(model)
local file = io.open("Count CSV/id_vector_train.csv")
l=0
for line in file:lines() do
	l=l+1
	A = line:split(",")
	--data[l] = torch.Tensor(A)
	--print(l,#A)
end
feature_size = #A
print(feature_size)
data = torch.Tensor(l,feature_size-1)
labels = torch.Tensor(l)

--open csv file
local file = io.open("Count CSV/id_vector_train.csv")
l=0
for line in file:lines() do
	l=l+1
	local A = line:split(",")
	data[l] = torch.Tensor(A):index(1,torch.linspace(2,feature_size,feature_size-1):long())
	--print(l,#A)
end
data:resize(l,1,1,feature_size-1);
print(data:size(),data:min(),data:max())

local file = io.open("Count CSV/train_y_new.csv")
l=0
for line in file:lines() do
	l=l+1
	local A = line:split(",")
	labels[l]=A[2]+1;
	--print(l,#A)
end

trainPortion = 0.8

indices = torch.randperm(data:size(1)):long()
trainData = data:index(1,indices:sub(1,(trainPortion*indices:size(1))))
trainLabels = labels:index(1,indices:sub(1,(trainPortion*indices:size(1))))
testData = data:index(1,indices:sub((trainPortion*indices:size(1)+1), indices:size(1)))
testLabels = labels:index(1,indices:sub((trainPortion*indices:size(1)+1), indices:size(1)))

--normalize the data
trainData:add(-trainData:mean())
trainData:div(trainData:std())
print(trainData:mean(), trainData:std())
N=trainData:size(1)
local theta,gradTheta = model:getParameters()
criterion = nn.ClassNLLCriterion()

local x,y

local feval = function(params)
    if theta~=params then
        theta:copy(params)
    end
    gradTheta:zero()
    out = model:forward(x)
    --print(#x,#out,#y)
    local loss = criterion:forward(out,y)
    local gradLoss = criterion:backward(out,y)
    model:backward(x,gradLoss)
    return loss, gradTheta
end

--training is done batch wise
batchSize = 500

indices = torch.randperm(trainData:size(1)):long()
--trainData = trainData:index(1,indices)
--trainLabels = trainLabels:index(1,indices)

epochs = 15
print('Training Starting')
local optimParams = {learningRate = 0.01, learningRateDecay = 0.002}
local _,loss
local losses = {}
for epoch=1,epochs do
    collectgarbage()
    print('Epoch '..epoch..'/'..epochs)
    for n=1,N-batchSize, batchSize do
        x = trainData:narrow(1,n,batchSize)
        y = trainLabels:narrow(1,n,batchSize)
        --print(y)
        _,loss = optim.adam(feval,theta,optimParams)
        losses[#losses + 1] = loss[1]
    end
    local plots={{'Training Loss', torch.linspace(1,#losses,#losses), torch.Tensor(losses), '-'}}
    gnuplot.pngfigure('Training2.png')
    gnuplot.plot(table.unpack(plots))
    gnuplot.ylabel('Loss')
    gnuplot.xlabel('Batch #')
    gnuplot.plotflush()
    --permute training data
    indices = torch.randperm(trainData:size(1)):long()
    trainData = trainData:index(1,indices)
    trainLabels = trainLabels:index(1,indices)
end

torch.save('Linear.t7',model)

N = testData:size(1)
teSize = N

print('Testing accuracy')
testData:add(-testData:mean())
testData:div(testData:std())
correct = 0
class_perform = {0,0,0,0,0}
class_size = {0,0,0,0,0}
classes = {'Slovak','French','Spanish','German','Polish'}
for i=1,N do
    local groundtruth = testLabels[i]
    local example = torch.Tensor(1,1,93);
    example = testData[i]
    class_size[groundtruth] = class_size[groundtruth] +1
    local prediction = model:forward(example)
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    --print(#example,#indices)
    --print('ground '..groundtruth, indices[1])
    if groundtruth == indices[1] then
        correct = correct + 1
        class_perform[groundtruth] = class_perform[groundtruth] + 1
    end
    collectgarbage()
end
print("Overall correct " .. correct .. " percentage correct" .. (100*correct/teSize) .. " % ")
for i=1,#classes do
   print(classes[i], 100*class_perform[i]/class_size[i] .. " % ")
end
