# This implementation of NaiveBayes assumes only categorical varaibles

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
	otp = data.frame("id" = testX[,1])
	
	dataFull = x
	dataFull["class"] = y

	#print("the quantity below is the total amount of each of the features in the dataset")
	allXiFreq = apply(dataFull[-1], 2, function(z) as.data.frame(table(unlist(z))))
	for (class in unique(y)) {
	  print(class)
		dataFullClass = dataFull[dataFull$"class" == class,]
	
		#print("the quantity below is the total amount of each of the features with the label 'class' in the dataset")
		allXiClassFreq = apply(dataFullClass[-c(1,ncol(dataFullClass))],  2, function(z) as.data.frame(table(unlist(z))))

		#print("a function to calculate P(xi|y) with xi = z and y = class with normalization so that features we haven't seen received equal probability to each class")
		pxiy = function(vect, i) {
		  ## do the sapply for the unique categories and then copy, that should make it faster
		  print(i)
		  sortedCats = unique(sort(vect))
		  fr = rep(0, length(sortedCats))
		  sortedCats = setNames(fr, sortedCats)
		  uxiy = sapply(names(sortedCats), function(z) allXiClassFreq[[i]][allXiClassFreq[[i]][1] == z,]$Freq)
		  if(length(uxiy[[1]]) == 0) {uxiy = sortedCats}
		  xiy = sapply(vect, function(z) uxiy[as.character(z)])
		  idxiy = sapply(xiy, is.na)
		  xiy[unlist(idxiy)] = 0
		  xiy = setNames(xiy, NULL)
		  xiy = unlist(xiy)
		  uxi = sapply(names(sortedCats), function(z) allXiFreq[[i]][allXiFreq[[i]][1] == z,]$Freq)
		  if(length(uxi[[1]]) == 0) {uxi = sortedCats}
		  xi = sapply(vect, function(z) uxi[as.character(z)])
		  idxi = sapply(xi, is.na)
		  xi[unlist(idxi)] = 0
		  xi = setNames(xi, NULL)
		  xi = unlist(xi)
			return ((xiy + 1)/(xi + 1 + length(unique(y))))
		}

		#print("we calculate a single P(xi|y), then their product and finally, their product with the P(y)")
		beg = sapply(1:ncol(testX[-1]), function(i) pxiy(as.numeric(as.vector(t(testX[-1][i]))), i))
		#print("That's just a product IIP(xi|y)")
		mid = apply(beg, 1, prod)
		#print("to not to get lost -- mid is a vertical vector with as many rows as there are examples which gives IIP(xi) for each example")
		finProdClass = mid*freqs[freqs[1] == class,]$py
		#print(finProdClass)
		#print(class(finProdClass))

		otp = cbind(otp, finProdClass)
		names(otp)[which(unique(y) == class)+1] = class
	}

	#print("otp gives us a data frame with a column for example id and columns for probabilities of belonging to each class")
	ans = colnames(otp[-1])[apply(otp[-1],1,which.max)]
	# the command below will be needed in our particular case, where classes are numeric, not strings
	# ans = sapply(ans, function(x) as.numeric(x)]
	print(otp)
	print(py)
	return(cbind(otp[1], class= as.numeric(ans)))
}
			     
			     datx = read.csv("id_vector_train.csv")
daty = read.csv("train_set_y.csv")
daty = daty[daty[1][,] %in% datx[1][,],]
tesx = read.csv("id_vector_test.csv")

## Usuccesful tests
ids = seq(1:273829)
sizsam = 27382*2
set.seed(12345)
samp = sample(ids, sizsam)

sampDatx = datx[-samp,]
sampDaty = daty[-samp,][-1]

sampTesx = datx[samp,]
preds = NBayes(sampDatx[1:219065,], sampDaty[1:219065,], sampTesx[,])
