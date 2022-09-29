library(plotly)
library(ggplot2)
library(tidyverse)
library(rbin)
library(rdrobust)
library(rddensity)
library(rddtools)

## Visualizations ##

rides = data.table::fread('/Volumes/GabeTB/Data/chicago/ride-hailing-preprocessed.csv') %>%
  as_tibble()

# ------------------

# Proportion clean vs non-clean
ggplot(data=rides,
       aes(x=trip_miles, y=has_clean_fare_info)) +
  geom_point()

# ------------------

# We expect a linear relationship - except for some non-linearity as a result of latent variables (due to 2d plotting)
rides %>% 
  filter(has_clean_fare_info == TRUE) %>%
  ggplot(aes(x = trip_miles, y=fare)) +
  geom_point()

# We expect a linear relationship - except for some non-linearity as a result of latent variables (due to 2d plotting)
rides %>% 
  filter(has_clean_fare_info == TRUE) %>%
  ggplot(aes(x = trip_minutes, y=fare)) +
  geom_point()

# ------------------

# Fare Histogram  -- TO DO --

# Fare_Ratio Histogram -- TO DO --

# ------------------

# Geographic Density Plot - THIS DOES NOT WORK - although it would be pretty cool
fig <- rides
fig <- fig %>% 
  layout(
    type = 'densitymapbox',
    lat = ~pickup_centroid_latitude,
    lon = ~pickup_centroid_longitude,
    coloraxis = 'coloraxis',
    radius = 10
  )

# ------------------

# Surge multiplier - by more rounding discontinuies
rides %>% 
  filter(has_clean_fare_info == TRUE &
           (fare_ratio >= 1.15 & fare_ratio <= 2.0)) %>%
  ggplot(aes(x = fare_ratio, y = fare)) +
  geom_point(position = position_jitter()) +
  geom_vline(xintercept = c(1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95), color='red', size=0.5)

# Surge multiplier plotted by range
rides %>% 
  filter(has_clean_fare_info == TRUE &
           (fare_ratio >= 1.15 & fare_ratio <= 1.35)) %>%
  ggplot(aes(x = fare_ratio, y=fare)) +
  geom_point(position = position_jitter()) +
  geom_vline(xintercept = 1.25, color='red', size=0.5)

# ------------------

# Surge multiplier plotted by range according to other covariates
rides %>% 
  filter(has_clean_fare_info == TRUE &
           (fare_ratio >= 1.15 & fare_ratio <= 1.35)) %>%
  ggplot(aes(x = fare_ratio, y=trip_miles)) +
  geom_point(position = position_jitter()) +
  geom_vline(xintercept = 1.25, color='red', size=0.5)

# ------------------

# This labeling and coloring of points / axis IS NOT RIGHT
rides %>% 
  filter(has_clean_fare_info == TRUE &
           (fare_rounded >= 1.15 & fare_rounded <= 1.35)) %>%
  ggplot(aes(x=fare_ratio, y=fare, color=factor(fare_ratio>1.25))) +
  geom_point(position = position_jitter()) +
  geom_vline(xintercept=1.25) +
  theme_bw() +
  theme(legend.title=element_blank()) +
  scale_color_manual(labels = c('Control', 'Treatment'), values=c('black', 'brown')) +
  geom_smooth(method='lm') +
  ggtitle('Fare Price ~ Surge Multiplier Rounding Discontinuity')

# ------------------

# By Chicago geographic location
rides %>% 
  filter(has_clean_fare_info == TRUE &
           (fare_rounded >= 1.15 & fare_rounded <= 1.35)) %>%
  ggplot(aes(x = fare_ratio, y=fare)) +
  geom_point(position = position_jitter()) +
  geom_vline(xintercept = 1.25, color='red', size=0.5) +
  facet_wrap(~ pickup_side)

# ------------------

central = rides %>%
  filter(has_clean_fare_info == TRUE &
           (fare_ratio >= 1.15 & fare_ratio <= 1.35) &
           (pickup_side == 'central'))

# ------------------

# Central pickup area RD plot with evenly-spaced bins selected to mimic the underlying variability of the data
# and is implemented using spacings estimators
(rdplot(x=central$fare_ratio, y=central$fare, 
        c=1.25, 
        title = 'Central Chicago - RD Plot'))

# Alternative RD plot using evenly-spaced bins selected to trace out the underlying regression function 
# (i.e., using the IMSE-optimal selector), also implemented using spacings estimators
(rdplot(x=central$fare_ratio, y=central$fare, 
        c=1.25, 
        binselect='es',
        title = 'Central Chicago - RD Plot'))

# Same plot as directly above, but with more bins using scale function
(rdplot(x=central$fare_ratio, y=central$fare, 
        c=1.25, 
        binselect='es',
        scale = 5, ## the number of bins used is five times larger than the optimal choice
        title = 'Central Chicago - RD Plot'))

# Restrict x axis
(rdplot(y=central$fare, x=central$fare_ratio,
        c = 1.25,
        x.lim=c(1.24, 1.260), ## bws = bandwidth selection
        x.lab="Surge Multiplier",
        y.lab="Fare ~ Distance + Time + Forcing Variable"))






