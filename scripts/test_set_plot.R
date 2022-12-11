library(foreign)
library(sf)
library(tidyverse)
library(collapse)
library(countrycode)
library(giscoR)

#reading dataset
dataset=read.csv("data/processed/test_set_with_pred.csv")
dataset$ctr_code=countrycode(dataset$ctr,origin="dhs",destination="iso3c")
dataset=dataset[c("hv271","ctr_code","pred")]
#collapsing by country
dataset=collap(dataset,  ~ ctr_code, fmean) 

africa_simple <- gisco_get_countries(region = "Africa", resolution = "20") %>% st_crop(xmin = -20, xmax = 60, ymin = -40, ymax = 40)

#merging with geometries of countries
dataset=merge(dataset,africa_simple,
              by.x=c("ctr_code"),by.y=c("ISO3_CODE"), all.x=TRUE)   %>% st_as_sf


graphics.off()
pdf(file="data/results/heatmap_ground_truth.pdf")

ggplot(africa_simple) +
  geom_sf(fill=NA) +
  geom_sf(data=dataset, mapping = aes(fill = hv271 ), colour = 'black', size = 0.1) + 
  coord_sf(datum = NA) + 
  scale_fill_gradient(high = 'red', low = 'yellow', na.value = 'grey', name=NULL) + labs(title="Ground truth") +
  theme(panel.background = element_rect(fill = NA), legend.position = "bottom", plot.title = element_text(hjust = 0.4) ) 

dev.off()

pdf(file="data/results/heatmap_pred.pdf")

ggplot(africa_simple) +
  geom_sf(fill=NA) +
  geom_sf(data=dataset, mapping = aes(fill = pred ), colour = 'black', size = 0.1) + 
  coord_sf(datum = NA) + 
  scale_fill_gradient(high = 'red', low = 'yellow', na.value = 'grey', name=NULL) + labs(title="Predictions") +
  theme(panel.background = element_rect(fill = NA), legend.position = "bottom", plot.title = element_text(hjust = 0.4) ) 


dev.off()