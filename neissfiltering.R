library(dplyr)
setwd('/Users/ayaawwad/Documents/LLM_project')

neiss23 <- read_csv('neiss2023csv.csv')


# filter the data for the desired codes
# define the codes we are interested in filtering in 
codes <- c(1329, 3215, 3258,5022, 5023, 5024, 5025, 5033, 5035, 5040, 5042 )

neiss_filtered <- neiss23 %>% filter(Product_1 %in% codes | 
                                       Product_2 %in% codes |
                                     Product_3 %in% codes)

neiss23_filtered <- write_csv(neiss_filtered, file ='neiss_filtered' )
