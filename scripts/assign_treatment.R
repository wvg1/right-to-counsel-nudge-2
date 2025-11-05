#load packages
library(tidyverse)
library(writexl)

#set working directory as needed
setwd("C:/Users/wvg1/Documents/right-to-counsel-nudge-2")

#set seed for reproducibility
set.seed(651)

#function to assign treatment based on a starting case_ID and number of cases to add
generate_assignments <- function(start_id, n, week) {
  
  assignments <- tibble(
    case_ID = start_id:(start_id + n - 1),
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

