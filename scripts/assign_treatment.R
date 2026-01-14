#load packages
library(tidyverse)
library(writexl)

#set working directory as needed
setwd("C:/Users/wvg1/Documents/right-to-counsel-nudge-2")

#set seed for reproducibility
set.seed(651)

#function to assign treatment based on a starting household_ID and number of households to add
generate_assignments <- function(start_id, n, week) {
  
  assignments <- tibble(
    household_ID = start_id:(start_id + n - 1),
    treatment = sample(c(0, 1), n, replace = TRUE)
  )
  
  filename <- paste0("assignments_week_", week, ".xlsx")
  write_xlsx(assignments, filename)
  message("Exported ", n, " assignments to: ", filename)
  
  return(assignments)
}
#create week 1 assignments
generate_assignments(start_id = 1, n = 81, week = 1)

#create week 2 assignments
generate_assignments(start_id = 82, n = 66, week = 2)

#create week 3 assignments
generate_assignments(start_id = 148, n = 46, week = 3)

#create week 4 assignments
generate_assignments(start_id = 194, n = 76, week = 4)

#create week 5 assignments
generate_assignments(start_id = 270, n = 57, week = 5)

#create week 6 assignments
generate_assignments(start_id = 327, n = 83, week = 6)

#create week 7 assignments
generate_assignments(start_id = 410, n = 53, week = 7)

#create week 8 assignments
generate_assignments(start_id = 463, n = 65, week = 8)

#create week 9 assignments
generate_assignments(start_id = 528, n = 30, week = 9)

#create week 10 assignments
generate_assignments(start_id = 558, n = 111, week = 10)

#create week 11 assignments
generate_assignments(start_id = 669, n = 65, week = 11)

#create week 12 assignments
generate_assignments(start_id = 734, n = 83, week = 12)

#create week 13 assignments
generate_assignments(start_id = 817, n = 32, week = 13)

#create week 14 assignments
generate_assignments(start_id = 849, n = 61, week = 14)

#create week 15 assignments
generate_assignments(start_id = 910, n = 119, week = 15)
