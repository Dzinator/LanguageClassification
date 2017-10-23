require 'nn';
require 'image';
require 'optim';
require 'gnuplot';

model = torch.load('Linear.t7')
print(model)
local file = io.open("Count CSV/id_vector_test.csv")
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

--open csv files
local file = io.open("Count CSV/id_vector_test.csv")
l=0
for line in file:lines() do
	l=l+1
	local A = line:split(",")
	data[l] = torch.Tensor(A):index(1,torch.linspace(2,feature_size,feature_size-1):long())
	--print(l,#A)
end
data:resize(l,1,1,feature_size-1);
print(data:size(),data:min(),data:max())

indices = torch.linspace(1,l,l):long()
testData = data:index(1,indices)
testLabels = labels:index(1,indices)

--normalize the data
testData:add(-testData:mean())
testData:div(testData:std())
print(testData:mean(), testData:std())
N=testData:size(1)
teSize = N

print('Testing accuracy')
--create file to write the predicted classes
local write_file = io.open("submission_AS.csv","w")
correct = 0
class_perform = {0,0,0,0,0}
class_size = {0,0,0,0,0}
classes = {'Slovak','French','Spanish','German','Polish'}
for i=1,N do
    local example = torch.Tensor(1,93);
    example = testData[i]
    local prediction = model:forward(example)
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    --print(#example,#indices)
    --print('ground '..groundtruth, indices[1])
    testLabels[i] = indices[1]-1;
    write_file:write(i-1);
    write_file:write(",");
    write_file:write(testLabels[i]);
    write_file:write("\n");
    collectgarbage()
end

