# This implementation of NaiveBayes assumes only categorical varaibles

NBayes = function(x,y, testX) {
	# x is a data.frame with the data
	# y is the class vector
	# testX is what we want to predict
	# the function returns predictions for testX

	# First we get the P(y) term

	freqs = as.data.frame(table(y))
	py = sapply(freqs[,2], function(x) x/length(y))
	freqs["py"] = py

	# Then, for each row and each class we find P(y|x) ~ P(x|y)P(x) ~ IIP(x_i|y)P(y)

	# here we'll store all the probabilities for each point in testX belonging to some class
	otp = data.frame("id" = testX[,1])
	
	dataFull = x
	dataFull["class"] = y

	# the quantity below is the total amount of each of the features in the dataset
	allXiFreq = apply(dataFull[-1], 2, function(z) as.data.frame(table(unlist(z))))
	for (class in unique(y)) {
		dataFullClass = dataFull[dataFull$"class" == class,]
	
		# the quantity below is the total amount of each of the features with the label 'class' in the dataset 
		allXiClassFreq = apply(dataFullClass[-c(1,ncol(dataFullClass))],  2, function(z) as.data.frame(table(unlist(z))))

		# a function to calculate P(xi|y) with xi = z and y = class with normalization so that features we haven't seen received equal probability to each class
		pxiy = function(vect, i) {
		  xiy = sapply(vect, function(z) allXiClassFreq[[i]][allXiClassFreq[[i]][1] == z,]$Freq)
		  idxiy = !(sapply(xiy, length))
		  xiy[idxiy] = 0
		  xiy = unlist(xiy)
		  xi = sapply(vect, function(z) allXiFreq[[i]][allXiFreq[[i]][1] == z,]$Freq)
		  idxi = !(sapply(xiy, length))
		  xi[idxi] = 0
		  xi = unlist(xi)
			return ((xiy + 1)/(xi + length(unique(y))))
		}

		# we calculate a single P(xi|y), then their product and finally, their product with the P(y)
		beg = sapply(1:ncol(testX[-1]), function(i) pxiy(as.numeric(as.vector(t(testX[-1][i]))), i))
		mid = apply(beg, 1, prod)
		## to not to get lost -- mid is a vertical vector with as many rows as there are examples which gives IIP(xi) for each example
		finProdClass = mid*freqs[freqs[1] == class,]$py

		otp = cbind(otp, finProdClass)
		names(otp)[which(unique(y) == class)+1] = class
	}

	# otp gives us a data frame with a column for example id and columns for probabilities of belonging to each class
	print(otp)
	ans = colnames(otp[-1])[apply(otp[-1],1,which.max)]
	print(ans)
	# the command below will be needed in our particular case, where classes are numeric, not strings
	# ans = sapply(ans, function(x) as.numeric(x)]
	return(cbind(otp[1], class= as.numeric(ans)))
}

## NBayes(datx, daty, tesx)
