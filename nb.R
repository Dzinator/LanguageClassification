# This implementation of NaiveBayes assumes either categorical or continuous variables (the code has to be modified a tiny bit to use one or the other)

## turn off "e" notation
options(scipen=999)

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
	#it's useful with categorical variables
	#allXiFreq = apply(dataFull[-1], 2, function(z) as.data.frame(table(unlist(z))))
	intgr = function(i, z) {
	  res = allXiFreq[[i]][allXiFreq[[i]][1] == z,]$Freq
	  if(length(res) == 0)
	    return(0)
	  return(res)
	}
	
	#this function implements Kullback-Leibler divergence measure to all the features
	getWeights = function() {
	  kull_leibsPOS = list()
	  kull_leibsNEG = list()
	  for(class in unique(y)) {
	    dataFullClass = dataFull[dataFull$"class" == class,]
	    y_and_xi = colSums(dataFullClass[-c(1,ncol(dataFullClass))])
	    xi = colSums(dataFull[-c(1,ncol(dataFullClass))])
	    p_yxi = (y_and_xi + 1)/(xi + 1 + unique(y))
	    #for K-L to be positive we need to have a distribution on features' categories
	    #here, we assume specifically for the problem in question, that there are 2 results: a feature can have frequency larger than 0 or 0
	    p_yNOTxi = 1 - p_yxi
	    p_y = freqs[freqs[1] == class,]$py
	    kullback_leiblerPOS = sapply(p_yxi, function(z) z*log2(z/p_y))
	    kullback_leiblerNEG = sapply(p_yNOTxi, function(z) if(z==0){0} else{z*log2(z/p_y)})
	    #print(kullback_leiblerPOS)
	    #print(kullback_leiblerNEG)
	    kull_leibsPOS[[as.character(class)]] = kullback_leiblerPOS
	    kull_leibsNEG[[as.character(class)]] = kullback_leiblerNEG
	  }
	  #a vector with a kullback-leibler measure for each feature
	  klPOS = Reduce("+", kull_leibsPOS)
	  klNEG = Reduce("+", kull_leibsNEG)
	  p_xi = colSums(dataFull[-c(1,ncol(dataFull))])/sum(colSums(dataFull[-c(1,ncol(dataFull))]))
	  p_NOTxi = 1 - p_xi
	  raw_weights = p_xi*klPOS + p_NOTxi*klNEG
	  normalizator = sum(raw_weights)/length(raw_weights)
	  weights = raw_weights/normalizator
	  return(weights)
	}
	
	weights = getWeights()
	
	for (class in unique(y)) {
	  print(class)
		dataFullClass = dataFull[dataFull$"class" == class,]
		yi = length(t(dataFullClass[1]))
		#for count data
		sumsAll = sum(colSums(dataFullClass[-c(1,ncol(dataFullClass))]))
	
		#print("the quantity below is the total amount of each of the features with the label 'class' in the dataset")
		#allXiClassFreq = apply(dataFullClass[-c(1,ncol(dataFullClass))],  2, function(z) as.data.frame(table(unlist(z))))
		intgrClass = function(i, z) {
		  res = allXiClassFreq[[i]][allXiClassFreq[[i]][1] == z,]$Freq
		  if(length(res) == 0)
		    return(0)
		  return(res)
		}
		#for count data
		sumsXi = apply(dataFullClass[-c(1,ncol(dataFullClass))],  2, function(z) sum(z))
		sumsXi = as.vector(t(sumsXi))

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
		
		#a function to calcualte P(xi|y) for count data
		pxiy3 = function(vect, i) {
		  print(i)
		  w = weights[i]
		  prob = (sumsXi[i] + 1)/(sumsAll + 1 + length(unique(y)))
		  probs = sapply(vect, function(z) if(z == 0) {1} else {(prob^z)^w})
		  return (probs)
		}

		#print("we calculate a single P(xi|y), then their product and finally, their product with the P(y)")
		beg = sapply(1:ncol(testX[-1]), function(i) pxiy3(as.numeric(as.vector(t(testX[-1][i]))), i))
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

## genetic algorithm to optimize the weights
fitBayes = function(w, x,y, testX) {
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
  #allXiFreq = apply(dataFull[-1], 2, function(z) as.data.frame(table(unlist(z))))
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
    sumsAll = sum(colSums(dataFullClass[-c(1,ncol(dataFullClass))]))
    
    #print("the quantity below is the total amount of each of the features with the label 'class' in the dataset")
    #allXiClassFreq = apply(dataFullClass[-c(1,ncol(dataFullClass))],  2, function(z) as.data.frame(table(unlist(z))))
    intgrClass = function(i, z) {
      res = allXiClassFreq[[i]][allXiClassFreq[[i]][1] == z,]$Freq
      if(length(res) == 0)
        return(0)
      return(res)
    }
    sumsXi = apply(dataFullClass[-c(1,ncol(dataFullClass))],  2, function(z) sum(z))
    sumsXi = as.vector(t(sumsXi))
    
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
    
    pxiy3 = function(vect, i) {
      prob = (w[i]*sumsXi[i] + 1)/(w[i]*sumsAll + 1 + length(unique(y)))
      probs = sapply(vect, function(z) if(z == 0) {1} else {(prob^z)^w[i]})
      return (probs)
    }
    
    #print("we calculate a single P(xi|y), then their product and finally, their product with the P(y)")
    beg = sapply(1:ncol(testX[-1]), function(i) pxiy3(as.numeric(as.vector(t(testX[-1][i]))), i))
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
  ret = cbind(otp[1], Category= as.numeric(ans))
  print(length(which(t(ret[2]) == sampTesy))/length(sampTesy))
  return(length(which(t(ret[2]) == sampTesy))/length(sampTesy))
}
wgt = NBayes(sampDatx, sampDaty, sampTesx) ## the code was changed to return just the weights
minn = min(wgt)
maxx = max(wgt)
sd = sd(wgt)
inputMin = sapply(wgt, function(x) if(x >= sd) x-sd else(minn))
inputMax = sapply(wgt, function(x) x+sd)
library(GA)
bestW = ga(type = "real-valued", fitness = fitBayes, sampDatx, sampDaty, sampTesx, min = rep(0,161), max = inputMax, popSize = 100, maxiter = 100, run = 30, keepBest = TRUE)

## read the data
datx = read.csv("id_vector_train_BIG.csv", header = FALSE)
daty = read.csv("train_y_new.csv", header = FALSE)
daty = daty[daty[1][,] %in% datx[1][,],]
names(daty) = c("ID", "Cat")
daty = as.vector(t(daty[-1]))
tesx = read.csv("id_vector_test_BIG.csv", header = FALSE)
rem = intersect(which(colSums(tesx) < 20), which(colSums(datx)<20))
remDatx = datx[-rem]
remTesx = tesx[-rem]

## Tests
ids = seq(1:273830)
sizsam = 27383*2
set.seed(12345)
samp = sample(ids, sizsam)
samp = sort(samp)

sampDatx = remDatx[-samp,]
sampDaty = daty[-samp]

sampTesx = remDatx[samp,]
sampTesy = daty[samp]
preds = NBayes(sampDatx, sampDaty, sampTesx)
print(length(which(t(preds[2]) == sampTesy))/length(sampTesy))

## Maybe cross-validation (baaad results)
library(caret)
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
write.csv(realPreds, "/home/julek/Desktop/nb_preds10.csv", row.names = FALSE)

## InfGain did not provide better results...
library(CORElearn)
IG = attrEval(Cat~., cbind(remDatx, daty[2])[-1], estimator = "InfGain")
GR = attrEval(Cat~., cbind(remDatx, daty[2])[-1], estimator = "GainRatio")
valuable = (IG > 0.01)
valuable = (GR > 0.05)
valDatx = remDatx[valuable]
valTesx = remTesx[valuable]

ids = seq(1:273830)
sizsam = 27383*2
set.seed(12345)
samp = sample(ids, sizsam)
samp = sort(samp)

valSampDatx = valDatx[-samp,]
valSampDaty = daty[-samp]

valSampTesx = valDatx[samp,]
valSampTesy = daty[samp]

valPreds = NBayes(valSampDatx, valSampDaty, valSampTesx)
print(length(which(t(valPreds[2]) == valSampTesy))/length(valSampTesy))

valRealPreds = NBayes(remDatx, daty, remTesx)
write.csv(valRealPreds, "/home/julek/Desktop/nb_preds7.csv", row.names = FALSE)

