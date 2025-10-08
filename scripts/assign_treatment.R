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


