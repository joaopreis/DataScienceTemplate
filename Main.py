import TrainingModels
from DataProfiling import *
from Evaluation import *
from DataPreparation import *

#############################################PROJECT DATA PROFILING#####################################################

data = readCSV("covtype.csv")
##data.drop(["Soil_Type1","Soil_Type2","Soil_Type3","Soil_Type4","Soil_Type5","Soil_Type6","Soil_Type7",


##print(data["Soil_Type7"])
##np.random.seed(2)
##sample = sampleData(data, 0.2)
print(dataShape(data))
print(dataType(data))
##print(describeData(sample))
##HistDist(sample, 5, 3, 0, 0)
##HistDist(data, 5, 3, 28, 28)
##HistDist(sample, 5, 3, 2, 2)
##HistDist(sample, 5, 3, 3, 3)
##HistDist(sample, 5, 3, 4, 4)
##HistDist(sample, 5, 3, 5, 5)
##HistDist(sample, 5, 3, 6, 6)
##HistDist(sample, 5, 3, 7, 7)
##HistDist(sample, 5, 3, 8, 8)
##HistDist(sample, 5, 3, 9, 9)
##nullValues(data, 10, 8)
##indBoxplot(sample, 4, 4, 0, 9)
##indBoxplot(sample, 4, 4, 10, 13)
##indBoxplot(sample, 2, 2, 14, 33)
##indBoxplot(sample, 2, 2, 35, 53)
##indBoxplot(sample, 4, 4, 54, 54)
##indNumHistogram(sample, 5, 3, 0, 9)
##indSnsHistogram(sample, 5, 3, 0, 9)
##indNumHistogram(data, 4, 4, 28, 28)
##indSnsHistogram(data, 4, 4, 10, 13)
##indNumHistogram(data, 4, 2, 14, 33)
##indSnsHistogram(data, 4, 2, 13, 33)
##indNumHistogram(sample, 4, 2, 35, 53)
##indSnsHistogram(data, 4, 4, 34, 53)
##indNumHistogram(data, 4, 2, 54, 54)
##indSnsHistogram(data, 4, 2, 54, 54)
##sparsity(sample.iloc[:, [1, 8]])
##sparsity(sample.iloc[:, [3, 4]])
##sparsity(sample.iloc[:, [7, 8]])
##heatMap(sample, 2, 2, 0, 9)

########################################################################################################################

###########################################Project training models######################################################


##naiveBayes(data, 0.7, "Cover_Type", GaussianNB())
##accuracy(data, 0.9, "Cover_Type", GaussianNB)
##confusionMatrix(data, 0.8, "Cover_Type", GaussianNB())
naiveBayesComparator(data, 0.70, "Cover_Type")
##kNNComparator(sample, 0.7, "Cover_Type")
##decisionTrees(sample, 0.7, "Cover_Type")
##randomForests(sample, 0.7, "Cover_Type")
##rocChart(sample, 0.7, "Cover_Type", GaussianNB())
