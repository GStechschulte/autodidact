library(ggplot2)
library(tidyverse)
library(rbin)
library(rdrobust)
library(rddensity)
library(rddtools)

## Regression Discontinuity Design (RDD)

rides = data.table::fread('/Volumes/GabeTB/Data/chicago/ride-hailing-preprocessed.csv') %>%
  as_tibble()

# ------------------

# Chicago-central
central = rides %>%
  filter(has_central_fare_info == TRUE &
           (fare_ratio >= 1.15 & fare_ratio <= 1.35) &
           (pickup_side == 'central'))

# ------------------

# Continuity-based analysis

# Density test
rides_rd_test <- rddensity(X = central$fare_ratio, c=1.25)
summary(rides_rd_test)

# the T-value
as.numeric(rides_rd_test$hat[3])/as.numeric(rides_rd_test$sd_jk[3])

# Histogram
ggplot(central, aes(x=fare_ratio, 
                  fill = factor(fare_ratio > 1.25))) +
  geom_histogram(binwidth=1) +
  xlim(1.15, 1.35) +
  geom_vline(xintercept = 1.25) +
  xlab("Surge Multiplier") +
  scale_fill_manual(values = c("black","brown"))+
  theme_bw() +
  theme(legend.position='none')

# ------------------

# Estimation of RD main effect
set.seed(42)

fit <- rdrobust(y=central$fare, x=central$fare_ratio, 
                all=TRUE,
                c=1.25)
summary(fit)


