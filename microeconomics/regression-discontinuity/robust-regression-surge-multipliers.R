source("/Users/gabestechschulte/Documents/git-repos/Side_Projects/regression-discontinuity/helpers.R")

library(ggplot2)
library(fasttime)
library(MASS)
library(lubridate)
library(tidyverse)
library(dplyr)
library(plotly)

# -----------------

tnp_trips = data.table::fread('/Volumes/GabeTB/Data/chicago/chicago-ride-hailing-2M.csv') %>%
  as_tibble() %>%
  filter(
    !is.na(pickup_community_area)
  )

# -----------------

## New dataframe with new features engineered and features with criteria filtering ##

tnp_trips = tnp_trips %>%
  # mutate is creating new columns
  mutate(
    trip_minutes = trip_seconds / 60,
    trip_start = fastPOSIXct(trip_start_timestamp, tz = "UTC"),
    pickup_census_tract = as.character(pickup_census_tract),
    dropoff_census_tract = as.character(dropoff_census_tract),
    trip_id = row_number(),
    share_requested = shared_trip_authorized %in% c("true", "false"),
    calendar_month = as.Date(floor_date(trip_start, unit = "month")),
    mph = 60 * trip_miles / trip_minutes,
    # Creates a new column "has_clean_fare_info" indicating "true or false" if this particular row (sample) meets the following criteria
    has_clean_fare_info = (
      !is.na(fare) & !is.na(trip_miles) & !is.na(trip_minutes) &
        (trip_miles >= 1.5 | trip_minutes >= 8) &
        trip_miles >= 0.5 & trip_miles < 100 &
        trip_minutes >= 2 & trip_minutes < 180 &
        mph >= 0.5 & mph < 80 &
        fare > 2 & fare < 1000 &
        fare / trip_miles < 25
    ) 
  ) %>%
  #select(-trip_seconds) %>%
  left_join(area_sides, by = c("pickup_community_area" = "community_area")) %>%
  rename(pickup_side = side)

# -----------------

## Baseline Fare Price w/Robust Regression - Modeling no surge-multiplier ##

set.seed(42)

# Variable where the model's coefficents will be
model_coefs = tnp_trips %>%
  distinct(share_requested) %>%
  arrange(share_requested) %>%
  mutate(coefs = list(NA))

fitted_values = tnp_trips %>%
  filter(has_clean_fare_info) %>%
  group_by(share_requested) %>%
  group_modify(~ {
    formula = if (.y$share_requested) {
      fare ~ (trip_miles + trip_minutes) * shared_status
    } else {
      fare ~ trip_miles + trip_minutes
    }
    
    model = MASS::rlm(formula, data = .x, method = "MM", init = "lts")
    
    fitted_values = modelr::add_predictions(.x, model, var = "predicted_fare") %>%
      select(trip_id, predicted_fare)
    
    model_coefs <<- model_coefs %>%
      mutate(coefs = case_when(
        (share_requested == .y$share_requested) ~ list(broom::tidy(model)),
        TRUE ~ coefs
      ))
    
    fitted_values
  }) %>%
  ungroup() %>%
  select(trip_id, predicted_fare)

# -----------------

tnp_trips = tnp_trips %>%
  left_join(fitted_values, by = "trip_id") %>%
  mutate(fare_ratio = fare / predicted_fare)

rm(fitted_values)

# -----------------

# Surge multiplier rounding discontinuity 
tnp_trips = tnp_trips %>%
  mutate(
    fare_rounded = round(fare_ratio, digits=1), ## 1.249 --> 1.2 or 1.251 --> 1.3
    fare_rounded = if_else(fare_rounded <= 1.0, 1, fare_rounded)
    )

# -----------------

# Consumer surplus resulting from rounding discontinuity
tnp_trips = tnp_trips %>%
  mutate(
    consumer_surplus = if_else(fare_rounded < fare_ratio, )
  )

# -----------------

# Write finished data.table out as a new 'csv' file
write.csv(tnp_trips,  file='/Volumes/GabeTB/Data/chicago/ride-hailing-preprocessed.csv', na='NA',
            row.names=FALSE)






