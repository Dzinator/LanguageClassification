require 'nn';

model = nn.Sequential()

--4 hidden layers, 93 features, 5 classes 

model:add(nn.Linear(93,256)) --1
model:add(nn.ReLU()) --2
model:add(nn.Dropout(0.5)) --3
model:add(nn.Linear(256,512)) --4
model:add(nn.ReLU()) --5
model:add(nn.Dropout(0.5)) --6
model:add(nn.Linear(512,128)) --7
model:add(nn.ReLU()) --8
model:add(nn.Dropout(0.5)) --9
model:add(nn.Linear(128,32)) --10
model:add(nn.ReLU()) --11
model:add(nn.Dropout(0.5)) --12
model:add(nn.Linear(32,5)) --13
model:add(nn.LogSoftMax())-- 14

return model