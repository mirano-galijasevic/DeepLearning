library(keras)
library(ggplot2)
library(tidyverse)
library(caret)

# set file name and working directory (change this to your paths)
file <- "adult.data"
setwd("C:\\Users\\mirano\\Documents\\R")

# download the file if it does not exists
if( !file.exists( file ))
{ 
  download.file( "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", file)
}

# load file into data.frame, and mark the NA strings...there are around 2500 missing values
data              <- read.table( file, sep = ",", header = FALSE, na.strings = " ?" )
nrows             <- nrow( data )
print(paste( "Data frame number of rows is: ", nrows ))

# let us now remove the missing values, and re-enumerate the rows
data              <- na.omit( data )
nrows             <- nrow( data )
row.names( data ) <- 1:nrows
print( paste( "After cleaning, the number of rows is: ", nrows ))

# set column names, can be found in adult.names file in the repo
colnames( data ) <- c( "age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", 
                        "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income" )

# separate text and number columns
train_text        <- data[,1:14] %>% select_if( is.character )
train_numbers     <- data[,1:14] %>% select_if( is.integer )

# do one-hot encoding of text values
encoded           <- caret::dummyVars( " ~ .", data = train_text )
train_encoded     <- data.frame( predict( encoded, newdata = train_text ))

# add these columns back into the main sets
label_var         <- data %>% select( income )
data              <- cbind( train_numbers, train_encoded, label_var )

# split data in 80%/20%
sample_size       <- floor( .8 * nrows )
set.seed(123)
train_idx         <- sample( nrows, size = sample_size )

# create training and test set
training_set      <- data[ train_idx, ]
test_set          <- data[ -train_idx, ]

# get label from training set, decrease it by 1, then remove it from training set
training_labels   <- as.numeric( factor(training_set[,15]) )
training_labels   <- training_labels - 1;
training_set      <- training_set %>% select( -income )

# get label from test set, decrease it by 1, then remove it from test set
test_labels       <- as.numeric( factor(test_set[,15]) )
test_labels       <- test_labels - 1;
test_set          <- test_set %>% select( -income )

# we need to convert the train and test data from data.frame to matrix
training_set      <- as.matrix( training_set )
test_set          <- as.matrix( test_set )

#create model consisting of linear stack of layers
model             <- keras_model_sequential()
model %>%
  layer_dense( name = 'DeepLayer01', units = 64, activation = 'relu', input_shape = ncol( training_set ) ) %>%
  layer_dropout( rate = 0.2) %>%
  layer_dense( name = 'DeepLayer02', units = 64, activation = 'relu' ) %>%
  layer_dropout( rate = 0.2 ) %>%
  layer_dense( name = 'OutputLayer', units = 1, activation = 'sigmoid' )

# compile model
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_sgd( lr = 0.01, momentum = 0, decay = 0 ),
  metrics = 'accuracy')

# number of epoch
epochs <- 100

# set callback
print_callback <- callback_lambda(
  on_epoch_end = function( e,l )
  {
    if ( e %% 80 == 0)
      cat( "\n" )
    cat( "." )
  }
)

stop_callback = callback_early_stopping( monitor = "accuracy", min_delta = 0, patience = 10, verbose = 0, mode = "auto" )

# train the model
stats <- model %>% fit(
  training_set,
  training_labels,
  epochs = epochs,
  batch = 16,
  validation_split = .15,
  verbose = 1,
  callbacks = list( 
      print_callback,
      stop_callback
    )
)

# evaluate the model against the test set
model %>%
  keras::evaluate( test_set, test_labels )

