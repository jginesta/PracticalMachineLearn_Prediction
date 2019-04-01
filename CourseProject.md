Practical Machine Learning - Week 4
================
Jessica Ginesta
March 31, 2019

Synopsis:
=========

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: <http://groupware.les.inf.puc-rio.br/har>

The goal of this project is to predict the manner in which the participants did the exercise.

Data Processing
---------------

Let's start by downloading and extracting only the fields we will use for this analysis.

``` r
#Load dataset
training <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"),header=TRUE)

testing <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"),header=TRUE)

#Cleaning data
# Remove variables that don't make intuitive sense for prediction which are the first 7.
trainClean<- training[, colSums(is.na(training)) == 0]
trainClean<-trainClean[,-c(1:7)]
testClean <- testing[, colSums(is.na(testing)) == 0]
testClean<-testClean[,-c(1:7)]

# Create a partition with the training dataset 
inTrain <- createDataPartition(trainClean$classe, p = 0.7, list = FALSE)
TrainSet <- trainClean[inTrain, ]
TestSet <- trainClean[-inTrain, ]

zeroVar <- nearZeroVar(TrainSet)
TrainSet <- TrainSet[, -zeroVar]
TestSet  <- TestSet[,-zeroVar]
dim(TrainSet)
dim(TestSet)
```

Initially loading the data, the testing dataset had 20 observations with 160 variables and the training set had 19622 observations with 160 observations. Looking at the data there were many variables that had NA values in most of their observations. We also removed the first 7 columns as it is personal data and would interfere in the predictions.

After creatinga partition with the test and training set we have: 1. New Training Set: 13737 observations with 53 variables. 2. New Testing Set: 5885 observations with 53 variables.

### Analyzing Data

We will run our test and train data through a couple of models to obtain the best accuracy. The number of folds is 5, as 10 is more resource intesive and doesn't not improve accuracy that much.

``` r
set.seed(1111)

# Fit classification tess as a model
# <- trainControl(method="cv", number=5)
trControl <- trainControl(method="cv", number=5)
modFit <- train(classe ~ .,method="rpart",data=na.omit(TrainSet),trControl=trControl)

# Print the classification tree
fancyRpartPlot(modFit$finalModel)
```

![](CourseProject_files/figure-markdown_github/unnamed-chunk-2-1.png)

``` r
predTree <- predict(modFit,newdata=TestSet)
tableTree <- confusionMatrix(TestSet$classe,predTree)
tableTree
accuracyTree<-tableTree$overall[1]
accuracyTree
```

The accuracy of the Tree is 0.5 which is not very good. Our next step will be trying the random forest.

### Random Forest

We will now fit the model on modelRandomF, and instruct the "train" function to use 3-fold cross-validation to select optimal tuning parameters for the model.

``` r
rfControl <- trainControl(method="cv", number=3,verboseIter=F)

modelRandomF <- train(classe~., data=na.omit(TrainSet), method="rf", trControl=rfControl)
print(modelRandomF)

# Print the accuracy
predRandomF <- predict(modelRandomF,newdata=TestSet)
tableRandomF <- confusionMatrix(TestSet$classe,predRandomF)
tableRandomF
accuracyRandomF<-tableRandomF$overall[1]
accuracyRandomF

plot(modelRandomF)
```

![](CourseProject_files/figure-markdown_github/unnamed-chunk-3-1.png) The accuracy of the random Forest is 0.99 which is great. Our next step will be trying to predict with the random forest model the Test data.

Conclusion
----------

Based on the accuracies above, the random forest model is the best one. We will now predict the values to use in the 20 questions test.

``` r
finalTest<-predict(modelRandomF,newdata=testClean)
finalTest
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
