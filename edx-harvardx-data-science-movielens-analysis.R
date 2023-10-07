##########################################################
# edX HarvardX: Data Science - MovieLens Capstone
# Analysis R Script
# Jason Katsaros
##########################################################

if (!require(renv))
  install.packages("renv", repos = "http://cran.us.r-project.org")
if (!require(tidyverse))
  install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(caret))
  install.packages("caret", repos = "http://cran.us.r-project.org")
if (!require(lubridate))
  install.packages("lubridate", repos = "http://cran.us.r-project.org")
if (!require(ggplot2))
  install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if (!require(recommenderlab))
  install.packages("recommenderlab", repos = "http://cran.us.r-project.org")

library(renv)
library(tidyverse)
library(caret)
library(lubridate)
library(ggplot2)
library(recommenderlab)

# Take a snapshot
renv::snapshot()

# Load in the edx data
base::load("rda/edx.rda")

# Function to calculate the Root Mean Squared Error (RMSE)
RMSE <- function(prediction, observed) {
  sqrt(mean((prediction - observed)^2))
}

# Visualize the general contents of the edx data set
edx %>% summarize(n_distinct(userId), n_distinct(movieId))
head(edx, 5)
summary(edx)

# Visualize the distribution of the ratings
edx %>%
  ggplot(aes(rating)) +
  geom_histogram()

# Get a list of unique genres across all movies
# To use in mutating the edx data set
edx %>% filter(!grepl("[|]", genres)) %>% distinct(genres) %>% arrange(genres)

# Clean up the edx data set before splitting it out
# into separate train and test sets
edx <- edx %>% mutate(
  date_reviewed = as_datetime(timestamp), # Convert the Unix time stamp to a date/time
  year_released = str_sub(title, -5, -2), # Extract the release year from the title of the movie
  title = str_sub(title, 0, -8), # Exclude the release year from the title of the movie
  action = grepl("Action", genres, ignore.case = TRUE), # Is this an "Action" movie?
  adventure = grepl("Adventure", genres, ignore.case = TRUE), # Is this an "Adventure" movie?
  children = grepl("Children", genres, ignore.case = TRUE), # Is this a "Children" movie?
  comedy = grepl("Comedy", genres, ignore.case = TRUE), # Is this a "Comedy" movie?
  crime = grepl("Crime", genres, ignore.case = TRUE), # Is this a "Crime" movie?
  documentary = grepl("Documentary", genres, ignore.case = TRUE), # Is this a "Documentary" movie?
  drama = grepl("Drama", genres, ignore.case = TRUE), # Is this a "Drama" movie?
  fantasy = grepl("Fantasy", genres, ignore.case = TRUE), # Is this a "Fantasy" movie?
  film_noir = grepl("Film-Noir", genres, ignore.case = TRUE), # Is this a "Film-Noir" movie?
  horror = grepl("Horror", genres, ignore.case = TRUE), # Is this a "Horror" movie?
  imax = grepl("IMAX", genres, ignore.case = TRUE), # Is this an "IMAX" movie?
  musical = grepl("Musical", genres, ignore.case = TRUE), # Is this a "Musical" movie?
  mystery = grepl("Mystery", genres, ignore.case = TRUE), # Is this a "Mystery" movie?
  romance = grepl("Romance", genres, ignore.case = TRUE), # Is this a "Romance" movie?
  sci_fi = grepl("Sci-Fi", genres, ignore.case = TRUE), # Is this a "Sci-Fi" movie?
  thriller = grepl("Thriller", genres, ignore.case = TRUE), # Is this a "Thriller" movie?
  war = grepl("War", genres, ignore.case = TRUE), # Is this a "War" movie?
  western = grepl("Western", genres, ignore.case = TRUE) # Is this a "Western" movie?
) %>%
  select(-c(timestamp, genres)) # Exclude "dirty" columns in favor of new columns

# Save the tidied edx data set
base::save(edx, file = "rda/edx.rda")

# Create training and testing partitions of the edx data
# These sets are separate from the final_holdout_test data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(edx$rating, times = 1, p = 0.7, list = FALSE)
train_set <- edx %>% slice(-test_index)
test_set <- edx %>% slice(test_index)

# Join the test set with the training set by both "movieId" and "userId"
# to ensure that the test set does not include movies and/or users that
# do not exist in the train set
test_set <- test_set %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Save the train_set and test_set data to RData files
base::save(train_set, file = "rda/train_set.rda")
base::save(test_set, file = "rda/test_set.rda")

# Naive Model
# Assume that all users rate all movies the same
# Take the average rating across all movies in the train set
naive_model <- mean(train_set$rating)
naive_model

# Compare the movie ratings in the test set against
# the average rating across all movies in the train set
naive_rmse <- caret::RMSE(test_set$rating, naive_model)
naive_rmse

# Print and save the RMSE to a results variable
rmse_results <- data.frame(method = "Naive Model", RMSE = naive_rmse)
rmse_results %>% knitr::kable()

# Save the results to a RData file
base::save(rmse_results, file = "rda/rmse_results.rda")

# Movie Effect Model
# Take the average rating across all movies in the train set
average_rating <- mean(train_set$rating)
average_rating

# Account for individual ratings against the average rating
# across all movies in the train set to create the movie model
movie_model <- train_set %>%
  group_by(movieId) %>%
  summarize(movie_average = mean(rating - average_rating))

# Join the movie model to the test set to predict
# the rating of movies in the test set
movie_predictions <- average_rating + test_set %>%
  left_join(movie_model, by = "movieId") %>%
  pull(movie_average)

# Compare the predicted ratings in the test set
# against the actual ratings in the test set
movie_rmse <- caret::RMSE(test_set$rating, movie_predictions)
movie_rmse

# Print and save the RMSE to a results variable
rmse_results <- bind_rows(
  rmse_results,
  data.frame(
    method = "Movie Effects Model",
    RMSE = movie_rmse
  )
)
rmse_results %>% knitr::kable()

# Save the results to a RData file
base::save(rmse_results, file = "rda/rmse_results.rda")

# Movie and User Effect Model
# Join the previously created movie model to the train set by "movieId"
# then find the average rating per user to create the user model
user_model <- train_set %>%
  left_join(movie_model, by = "movieId") %>%
  group_by(userId) %>%
  summarize(user_average = mean(rating - average_rating - movie_average))

# Join the movie model and the user model to the test set to predict
# the rating of movies in the test set
movie_and_user_predictions <- test_set %>%
  left_join(movie_model, by = "movieId") %>%
  left_join(user_model, by = "userId") %>%
  mutate(movie_and_user_prediction = average_rating + movie_average + user_average) %>%
  pull(movie_and_user_prediction)

# Compare the predicted ratings in the test set
# against the actual ratings in the test set
movie_and_user_rmse <- caret::RMSE(test_set$rating, movie_and_user_predictions)
movie_and_user_rmse

# Print and save the RMSE to a results variable
rmse_results <- bind_rows(
  rmse_results,
  data.frame(
    method = "Movie and User Effects Model",
    RMSE = movie_and_user_rmse
  )
)
rmse_results %>% knitr::kable()

# Save the results to a RData file
base::save(rmse_results, file = "rda/rmse_results.rda")

# Regularized Movie and User Effects Model

# Visualize the distribution of the number of ratings across movies
edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram() +
  scale_x_log10() +
  xlab("Number of Ratings") +
  ylab("Number of Movies") +
  ggtitle("Number of Ratings Per Movie")

# View a table of all movies with just one review
edx %>%
  group_by(movieId) %>%
  summarize(rating_count = n()) %>%
  filter(rating_count == 1) %>%
  left_join(edx, by = "movieId") %>%
  group_by(title) %>%
  summarize(rating = rating, rating_count = rating_count) %>%
  knitr::kable()

# Regularized Movie Effects Model
# Create the movie model again,
# but this time remove any movie that only has one review
regularized_movie_model <- train_set %>%
  group_by(movieId) %>%
  filter(n() > 1) %>%
  summarize(regularized_movie_average = mean(rating - average_rating))

# Filter the test set of the movies removed previously
filtered_test_set <- test_set %>%
  semi_join(regularized_movie_model, by = "movieId")

# Join the regularized movie model to the test set to predict
# the rating of movies in the test set
regularized_movie_predictions <- average_rating + filtered_test_set %>%
  left_join(regularized_movie_model, by = "movieId") %>%
  pull(regularized_movie_average)

# Compare the predicted ratings in the test set
# against the actual ratings in the test set
regularized_movie_rmse <- caret::RMSE(filtered_test_set$rating, regularized_movie_predictions)
regularized_movie_rmse

# Print and save the RMSE to a results variable
rmse_results <- bind_rows(
  rmse_results,
  data.frame(
    method = "Regularized Movie Effects Model",
    RMSE = regularized_movie_rmse
  )
)
rmse_results %>% knitr::kable()

# Save the results to a RData file
base::save(rmse_results, file = "rda/rmse_results.rda")

# Movie and Regularized User Effects Model
# Create the user model again,
# but this time remove any user with less than 100 movie reviews
regularized_user_model <- train_set %>%
  left_join(movie_model, by = "movieId") %>%
  group_by(userId) %>%
  filter(n() > 100) %>%
  summarize(regularized_user_average = mean(rating - average_rating - movie_average))
  
# Filter the test set of the users removed previously
filtered_test_set <- test_set %>%
  semi_join(regularized_user_model, by = "userId")

# Join the movie model and the regularized user model to the test set to predict
# the rating of movies in the test set
movie_and_regularized_user_predictions <- filtered_test_set %>%
  left_join(movie_model, by = "movieId") %>%
  left_join(regularized_user_model, by = "userId") %>%
  mutate(
    movie_and_regularized_user_prediction =
      average_rating +
      movie_average +
      regularized_user_average
  ) %>%
  pull(movie_and_regularized_user_prediction)

# Compare the predicted ratings in the test set
# against the actual ratings in the test set
# First we need to filter the test set of the users we removed previously
movie_and_regularized_user_rmse <- caret::RMSE(
  filtered_test_set$rating,
  movie_and_regularized_user_predictions
)
movie_and_regularized_user_rmse

# Print and save the RMSE to a results variable
rmse_results <- bind_rows(
  rmse_results,
  data.frame(
    method = "Movie and Regularized User Effects Model",
    RMSE = movie_and_regularized_user_rmse
  )
)
rmse_results %>% knitr::kable()

# Save the results to a RData file
base::save(rmse_results, file = "rda/rmse_results.rda")

# Regularized Model
# Combine the two regularized models above
# to create one regularized model
# Filter the test set of the movies and users removed previously
regularized_test_set <- test_set %>%
  semi_join(regularized_movie_model, by = "movieId") %>%
  semi_join(regularized_user_model, by = "userId")

# Join the regularized movie model and the regularized user model to the test set to predict
# the rating of movies in the test set
regularized_predictions <- regularized_test_set %>%
  left_join(regularized_movie_model, by = "movieId") %>%
  left_join(regularized_user_model, by = "userId") %>%
  mutate(
    regularized_prediction =
      average_rating +
      regularized_movie_average +
      regularized_user_average
  ) %>%
  pull(regularized_prediction)

# Compare the predicted ratings in the test set
# against the actual ratings in the test set
regularized_rmse <- caret::RMSE(regularized_test_set$rating, regularized_predictions)
regularized_rmse

# Print and save the RMSE to a results variable
rmse_results <- bind_rows(
  rmse_results,
  data.frame(
    method = "Regularized Model",
    RMSE = regularized_rmse
  )
)
rmse_results %>% knitr::kable()

# Save the results to a RData file
base::save(rmse_results, file = "rda/rmse_results.rda")

# Some movies recommendations should be weighted slightly seasonally
# as some users will prefer to watch seasonal movies at appropriate times of the year,
# while other users won't be affected by seasonally appropriate movies for a variety of reasons
# To test this relationship, here is a Chi-Squared test for Horror movies and October review dates
horror_october <- train_set %>%
  filter(month(date_reviewed) == "10" & horror == TRUE)
not_horror_october <- train_set %>%
  filter(month(date_reviewed) == "10" & horror == FALSE)
not_horror_not_october <- train_set %>%
  filter(month(date_reviewed) != "10" & horror == FALSE)
horror_not_october <- train_set %>% 
  filter(month(date_reviewed) != "10" & horror == TRUE)

horror_movie_matrix <- matrix(
  c(
    nrow(horror_october),
    nrow(not_horror_october),
    nrow(horror_not_october),
    nrow(not_horror_not_october)
  ),
  ncol = 2
)
colnames(horror_movie_matrix) <- c("October", "Not October")
rownames(horror_movie_matrix) <- c("Horror", "Not Horror")
horror_movie_matrix

chisq.test(horror_movie_matrix)

# Regularized With Seasonality Model
# Currently, the only seasonal genre that exists in the MovieLens data set is Horror
# A future enhancement could be to include a Holiday genre
regularized_seasonal_movie_model <- train_set %>%
  group_by(movieId) %>%
  filter(n() > 1) %>%
  summarize(
    regularized_movie_average =
      mean(rating -
             average_rating -
             ifelse(month(now()) == 10 &
                      horror, 0.125, 0
             )
      )
  )

# Filter the test set of the movies and users removed previously
regularized_seasonal_test_set <- test_set %>%
  semi_join(regularized_seasonal_movie_model, by = "movieId") %>%
  semi_join(regularized_user_model, by = "userId")

# Join the regularized movie model and the regularized user model to the test set to predict
# the rating of movies in the test set
regularized_seasonal_predictions <- regularized_seasonal_test_set %>%
  left_join(regularized_seasonal_movie_model, by = "movieId") %>%
  left_join(regularized_user_model, by = "userId") %>%
  mutate(
    regularized_seasonal_prediction =
      average_rating +
      regularized_movie_average +
      regularized_user_average
  ) %>%
  pull(regularized_seasonal_prediction)

# Compare the predicted ratings in the test set
# against the actual ratings in the test set
regularized_seasonal_rmse <- caret::RMSE(
  regularized_seasonal_test_set$rating,
  regularized_seasonal_predictions
)
regularized_seasonal_rmse

# Print and save the RMSE to a results variable
rmse_results <- bind_rows(
  rmse_results,
  data.frame(
    method = "Regularized With Seasonality Model",
    RMSE = regularized_seasonal_rmse
  )
)
rmse_results %>% knitr::kable()

# Save the results to a RData file
base::save(rmse_results, file = "rda/rmse_results.rda")

# RecommenderLab
# Convert a subset of the edx data set into a ratings matrix
# (for demonstration purposes)
edx_r <- as(edx[1:100000,], "realRatingMatrix")

# The proportion of items from the edx set to use for a train set
train_proportion <- 0.75

# This determines the number of items that should be given
# for evaluation when building the model
# The "given" parameter for evaluationScheme should not be larger than this value
min(rowCounts(edx_r))

# Define the "given" parameter for evaluationScheme
items_per_test_user_keep <- 10

# Define the threshold for a "good" movie rating
# (4 out of 5 stars is considered a "good" movie rating here)
good_threshold <- 4

set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
# Create the scheme used to train and evaluate the model
model_train_scheme <- edx_r %>%
  evaluationScheme(
    method = "split", # Splits the edx_r matrix into one train set and one test set
    train = train_proportion, # The proportion of the edx_r matrix to put into the train set
    given = items_per_test_user_keep, # The number of items to evaluate when building the model
    goodRating = good_threshold, # What is considered a "good" movie rating?
    k = 1
  )

# Create the parameters used to generate the model
model_parameters <- list(
  method = "cosine", # Use cosine similarity (https://en.wikipedia.org/wiki/Cosine_similarity)
  nn = 10, # Find each user's 1- most similar users in terms of preferences
  sample = FALSE, # The train set and test set are already created, so no need here
  normalize = "center"
)

# Describe User-based Collaborative Filtering
tail(recommenderRegistry$get_entries(dataType = "realRatingMatrix"), 1)

# Create the model
recommenderlab_model <- getData(
  model_train_scheme,
  "train"
) %>%
  Recommender(
    method = "UBCF", # User-based Collaborative Filtering
    parameter = model_parameters
  )

# Make predictions based on the model
recommenderlab_predictions <- recommenderlab::predict(
  recommenderlab_model,
  getData(
    model_train_scheme,
    "known" # Use what is known to predict the unknown portion (test set)
  ),
  type = "ratings"
)

# Evaluate the accuracy of the model
recommenderlab_error <- calcPredictionAccuracy(
  recommenderlab_predictions,
  getData(
    model_train_scheme,
    "unknown" # Use what is unknown to evaluate the accuracy of the model
  ),
  byUser = TRUE
)

# Compare the predicted ratings in the test set
# against the actual ratings in the test set
head(recommenderlab_error)

# Plot the RMSE density over all test set users
data.frame(
  user_id = as.numeric(
    row.names(recommenderlab_error)
  ),
  rmse = recommenderlab_error[, 1],
  predicted_items_count = rowCounts(
    getData(
      model_train_scheme,
      "unknown"
    )
  )
) %>%
  ggplot(aes(rmse)) +  
  geom_histogram(aes(y = ..density..)) +
  labs(title = 'UBCF RMSE on Predicted Recommendations Per Test User')

# Conclusion
# Load in the edx data
base::load("rda/final_holdout_test.rda")

# Filter the final holdout test set
# just like the regular test set
filtered_final_holdout_test <- final_holdout_test %>%
  semi_join(regularized_movie_model, by = "movieId") %>%
  semi_join(regularized_user_model, by = "userId")

# Join the regularized movie model
# and the regularized user model
# to the final holdout test set to predict
# the rating of movies in the final holdout test set
final_predictions <- filtered_final_holdout_test %>%
  left_join(regularized_movie_model, by = "movieId") %>%
  left_join(regularized_user_model, by = "userId") %>%
  mutate(
    final_prediction =
      average_rating +
      regularized_movie_average +
      regularized_user_average
  ) %>%
  pull(final_prediction)

# Compare the predicted ratings in the test set
# against the actual ratings in the test set
final_rmse <- caret::RMSE(filtered_final_holdout_test$rating, final_predictions)
final_rmse

# Print and save the naive RMSE to a results variable
rmse_results <- bind_rows(
  rmse_results,
  data.frame(
    method = "Final Model",
    RMSE = final_rmse
  )
)
rmse_results %>% knitr::kable()

# Save the results to a RData file
base::save(rmse_results, file = "rda/rmse_results.rda")