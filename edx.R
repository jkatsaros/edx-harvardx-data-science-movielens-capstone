##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if (!require(tidyverse))
  install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(caret))
  install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

# Download the zip file to the "data" directory
dl <- "data/ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# Extract the ratings data
ratings_file <- "data/ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, str_sub(ratings_file, 6), exdir = "data")

# Extract the movies data
movies_file <- "data/ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, str_sub(movies_file, 6), exdir = "data")

# Read the contents of the "ratings.dat" file into a data frame
ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
# Set the column names for the ratings data frame
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
# Set the data types of the information in the ratings data frame
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

# Read the contents of the "movies.dat" file into a data frame
movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
# Set the column names for the movies data frame
colnames(movies) <- c("movieId", "title", "genres")
# Set the data types of the information in the movies data frame
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

# Join the movies data frame to the ratings data frame by "movieId"
movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Save the edx data to an RData file
save(edx, file = "rda/edx.rda")

# Save the final_holdout_test data to an RData file
save(final_holdout_test, file = "rda/final_holdout_test.rda")
