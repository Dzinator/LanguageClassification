# This implementation of NaiveBayes assumes either categorical or continuous variables (the code has to be modified a tiny bit ti use one or the other)

NBayes = function(x,y, testX) {
	# x is a data.frame with the data
	# y is the class vector
	# testX is what we want to predict (data frame)
	# the function returns predictions for testX

	#print("First we get the P(y) term")

	freqs = as.data.frame(table(y))
	py = sapply(freqs[,2], function(x) x/length(y))
	freqs["py"] = py

	#print("Then, for each row and each class we find P(y|x) ~ P(x|y)P(x) ~ IIP(x_i|y)P(y)")

	#print("here we'll store all the probabilities for each point in testX belonging to some class")
	otp = data.frame("Id" = testX[,1])
	
	dataFull = x
	dataFull["class"] = y

	#print("the quantity below is the total amount of each of the features in the dataset")
	allXiFreq = apply(dataFull[-1], 2, function(z) as.data.frame(table(unlist(z))))
	intgr = function(i, z) {
	  res = allXiFreq[[i]][allXiFreq[[i]][1] == z,]$Freq
	  if(length(res) == 0)
	    return(0)
	  return(res)
	}
	
	for (class in unique(y)) {
	  print(class)
		dataFullClass = dataFull[dataFull$"class" == class,]
		yi = length(t(dataFullClass[1]))
	
		#print("the quantity below is the total amount of each of the features with the label 'class' in the dataset")
		allXiClassFreq = apply(dataFullClass[-c(1,ncol(dataFullClass))],  2, function(z) as.data.frame(table(unlist(z))))
		intgrClass = function(i, z) {
		  res = allXiClassFreq[[i]][allXiClassFreq[[i]][1] == z,]$Freq
		  if(length(res) == 0)
		    return(0)
		  return(res)
		}

		#print("a function to calculate P(xi|y) with xi = z and y = class with normalization so that features we haven't seen received equal probability to each class")
		pxiy = function(vect, i) {
		  ## do the sapply for the unique categories and then copy, that should make it faster
		  print(i)
		  sortedCats = unique(sort(vect))
		  fr = rep(0, length(sortedCats))
		  sortedCats = setNames(fr, sortedCats)
		  uxiy = sapply(names(sortedCats), function(z) intgrClass(i, z))
		  if(length(uxiy[[1]]) == 0) {uxiy = sortedCats}
		  xiy = sapply(vect, function(z) uxiy[as.character(z)])
		  idxiy = sapply(xiy, is.na)
		  xiy[idxiy] = 0
		  xiy = setNames(xiy, NULL)
		  xiy = unlist(xiy)
			return ((xiy + 1)/(yi + 1 + length(unique(y))))
		}
		  
		  ## print("a second function for continuous data; we assume each feature is normally distributed as use pdf to aproximate P(xi|y)")
		  allXiClassMeanSd = apply(dataFullClass[-c(1,ncol(dataFullClass))],  2, function(z) c(mean(z), sd(z)))

		  pxiy2 = function(vect, i) {
		    print(i)
		    mean = allXiClassMeanSd[1,i]
		    sd = allXiClassMeanSd[2,i]
		    probs = sapply(vect, function(z) pnorm(z, mean=mean, sd=sd))
		    return(probs)
		  }
		
	

		#print("we calculate a single P(xi|y), then their product and finally, their product with the P(y)")
		beg = sapply(1:ncol(testX[-1]), function(i) pxiy(as.numeric(as.vector(t(testX[-1][i]))), i))
		#print(beg)
		#print("That's just a product IIP(xi|y)")
		mid = apply(beg, 1, prod)
		#print("to not to get lost -- mid is a vertical vector with as many rows as there are examples which gives IIP(xi) for each example")
		finProdClass = mid*freqs[freqs[1] == class,]$py

		otp = cbind(otp, finProdClass)
		names(otp)[which(unique(y) == class)+1] = class
	}

	#print("otp gives us a data frame with a column for example id and columns for probabilities of belonging to each class")
	ans = colnames(otp[-1])[apply(otp[-1],1,which.max)]
	print(otp)
	print(py)
	return(cbind(otp[1], Category= as.numeric(ans)))
}

datx = read.csv("id_vector_train.csv", header = FALSE)
daty = read.csv("train_set_y.csv")
daty = daty[daty[1][,] %in% datx[1][,],]
daty = as.vector(t(daty[-1]))
tesx = read.csv("id_vector_test2.csv", header = FALSE)

## Tests
ids = seq(1:273830)
sizsam = 27382*2
set.seed(12345)
samp = sample(ids, sizsam)
samp = sort(samp)

sampDatx = datx[-samp,]
sampDaty = daty[-samp]

sampTesx = datx[samp,]
sampTesy = daty[samp]
preds = NBayes(sampDatx, sampDaty, sampTesx)
print(length(which(t(preds[2]) == sampTesy))/length(sampTesy))

preds2 = NBayes(sampDatx[,-c(2:26)], sampDaty, sampTesx[,-c(2:26)])
print(length(which(t(preds2[2]) == sampTesy))/length(sampTesy))

## Maybe cross-validation (baaad results)
otcm = c(1:length(daty))
flds = createFolds(otcm, k=10, list=TRUE, returnTrain=FALSE)
sumAcc = 0
allFlds = data.frame(Id = numeric(0), Category = numeric(0))
for(i in c(1:10)) {
  fldx = datx[-flds[[i]],]
  fldy = daty[-flds[[i]]]
  fldValx = datx[flds[[i]],]
  fldValy = daty[flds[[i]]]
  fldPreds = NBayes(fldx, fldy, fldValx)
  acc = length(which(fldPreds[2] == fldValy[2])) / length(t(fldPreds[2]))
  print(acc)
  sumAcc = sumAcc + acc
  allFlds = rbind(allFlds, fldPreds)
}
print(sumAcc/10)

## Real prediction
realPreds = NBayes(datx, daty, tesx)
write.csv(realPreds, "/home/julek/Desktop/nb_preds2.csv", row.names = FALSE)
