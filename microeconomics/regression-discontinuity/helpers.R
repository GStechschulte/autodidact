required_packages = c(
  'data.table',
  'fasttime',
  'lubridate',
  'MASS',
  'patchwork',
  'tidyverse',
  'zoo'
)

library(tidyverse)
library(lubridate)
library(patchwork)
library(zoo)
library(fasttime)

area_sides = tibble(
  community_area = 1:77,
  side = case_when(
    community_area %in% c(8, 32, 33) ~ "central",
    community_area %in% c(5:7, 21:22) ~ "north",
    community_area %in% c(1:4, 9:14, 77) ~ "far_north",
    community_area %in% 15:20 ~ "northwest",
    community_area %in% 23:31 ~ "west",
    community_area %in% c(34:43, 60, 69) ~ "south",
    community_area %in% c(57:59, 61:68) ~ "southwest",
    community_area %in% 44:55 ~ "far_southeast",
    community_area %in% 70:75 ~ "far_southwest",
    community_area %in% c(56, 76) ~ "airports",
  )
)

major_venues = list(
  soldier_field = "17031330100",
  united_center = "17031838100"
)
