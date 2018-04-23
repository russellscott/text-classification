
# analysis of textual information from multiple files, in multiple folders



# Libraries ---------------------------------------------------------------

pkgs <- c('readtext', 'tm', 'textstem', 'caret', 'naivebayes', 'plyr', 'rpart', 'ipred', 'e1071')

# Check if the package is installed. If not, install it.
newPkgs <- pkgs[!(pkgs %in% installed.packages()[, "Package"])]
if (length(newPkgs)) {install.packages(newPkgs, 
                                       repos="http://cran.rstudio.com/")}


# Load packages
lapply(pkgs, require, character.only = T)



# Load in data ------------------------------------------------------------


########## path: INPUT YOUR OWN PATH ###########
path = 'XXX'

# list of folders inside
list_folders_path = list.dirs(path = path, full.names = T)[-1]


# loop over folders and collect all text communications
all_docs = c()

for (i in 1:length(list_folders_path)){
  # read in data
  docs = readtext(list_folders_path[i])
  # add column indicating category
  docs$category = i
  docs = docs[,c(1,3,2)]
  
  doc_name = paste('doc', i, sep = '_')
  assign(doc_name, docs)
  rm(docs)
  # combine
  if (i == length(list_folders_path)){
    # combine
    all_docs = rbind(doc_1, doc_2, doc_3, doc_4, doc_5, doc_6, doc_7)
    rm(list = ls(pattern = 'doc_'))
  }
}


# clean up text -----------------------------------------------------------



preprocessing = function(all_docs){
  all_docs$text = tolower(all_docs$text)
  all_docs$text = removeWords(all_docs$text, words = stopwords('english'))
  all_docs$text = gsub('[[:punct:] ]+',' ',all_docs$text)
  all_docs$text = gsub('\n',' ', all_docs$text)
  all_docs$text = gsub('\\b\\w{1,2}\\b', '', all_docs$text)
  all_docs$text = stripWhitespace(all_docs$text)
  all_docs$text = lemmatize_words(all_docs$text)
  
  return(all_docs)
}

all_docs = preprocessing(all_docs)

# Corpus  ---------------------------------------------------------

#Build corpus
corpus = Corpus(VectorSource(all_docs$text))
#Get frequency of words
frequencies = DocumentTermMatrix(corpus)
#Dimension before removing sparse terms
dim(frequencies)
# Remove sparse terms
corpus_dense = removeSparseTerms(frequencies, 0.99)
#Dimension after removing sparse terms
dim(corpus_dense)

# Convert to a data frame
comm_df = as.data.frame(as.matrix(corpus_dense))

# Make all variable names R-friendly
colnames(comm_df) = make.names(colnames(comm_df))

# Add category variable
comm_df$category = factor(all_docs$category)
comm_df = comm_df[,c(dim(comm_df)[2], 1:(dim(comm_df)[2]-1))]
colnames(comm_df)=iconv(colnames(comm_df), to='ASCII', sub='')


# Modelling ---------------------------------------------------------------


# Split the data to test and train dataset
train.index = createDataPartition(comm_df$category, p = .8, list = FALSE)
train = comm_df[train.index,]
test = comm_df[-train.index,]

# shuffle rows in training and test sets
train = train[sample(1:nrow(train)), ]
test = test[sample(1:nrow(test)), ]

# setup model
predictors = 2:dim(comm_df)[2]
labels = 1
# 3 fold cross-validation
myControl <- trainControl(method='cv', number=3, returnResamp='none')


# build function for metrics
metrics_func = function(y_actual, y_predicted, title=NULL) {
  # confusion matrix
  cm = as.matrix(table(Actual = y_actual, Predicted = y_predicted)) 
  
  # metrics
  n = sum(cm) # number of instances
  nc = nrow(cm) # number of classes
  diag = diag(cm) # number of correctly classified instances per class 
  rowsums = apply(cm, 1, sum) # number of instances per class
  colsums = apply(cm, 2, sum) # number of predictions per class
  
  accuracy = sum(diag) / n 
  precision = diag / colsums 
  recall = diag / rowsums 
  f1 = 2 * precision * recall / (precision + recall)
  
  macroPrecision = mean(precision)
  macroRecall = mean(recall)
  macroF1 = mean(f1)
  
  return(data.frame(macroPrecision, macroRecall, macroF1, row.names = title))
}

# 3 models tested: Naive Bayes, SVM, and CART
# run models, predict, and save model and predictions
mod1 = caret::train(train[,predictors], train[,labels], method = 'naive_bayes', trControl=myControl)
pred1 = predict(object=mod1, test[,predictors])
#save(mod1, pred1, file='mod1.RData')

mod2 = caret::train(train[,predictors], train[,labels], method = 'svmRadialWeights', trControl=myControl)
pred2 = predict(object=mod2, test[,predictors])
#save(mod2, pred2, file='mod2.RData')

load('mod1.RData')
load('mod2.RData')
metrics_func(test[,labels], pred1, title = 'Naive Bayes')
metrics_func(test[,labels], pred2, title = 'SVM')
