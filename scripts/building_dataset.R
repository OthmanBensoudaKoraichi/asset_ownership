library(foreign)
library(readstata13)
library(sf)
library(tidyverse)
library(collapse)
library(countrycode)
library(giscoR)

setwd("data/DHS")

#checking for raw data
raw=unique((substr(list.files(),3,4)))


#dummy to mark if we have a dataframe to start appending stuff
start=0

#building a list of countries
ctr_list=unique((substr(list.files(),1,2)))
ctr_list=ctr_list[! ctr_list %in% c("DH","GP")]


#looping over countries to detect files
for (ctr in ctr_list) {
  print(paste("Country:",ctr))
  #finding all files for this country
  all_ctr=list.files(pattern=paste0("^",ctr,sep=""))
  #getting all versions of survey for that country
  vers=unique(as.integer(substr(all_ctr,5,6)))
  
  #looping over versions to find matches of data and gps
  for (ver in vers) {
    n_matches=0 #counter for number of matches
    
    #all zip files with a matching pattern
    zips=list.files(pattern=paste0("^",ctr,".*?",ver,sep=""))
    #if list of zips has length 2, means we have a match (survey with corresponding GPS data)
    if ( length(zips)==2  ) {
      print(paste("match found:",ctr,ver))
      n_matches=n_matches+1
      
      tryCatch( #trycatch, so we keep going when we find groups without data
      {
      #unzipping files temporarily
      for (zip in zips) {
        unzip(zip,exdir="unzipped")
      }
      
      #name of the survey
      data_name=list.files(path="unzipped",pattern=paste0("^",ctr,".*?",ver,".*?\\.DTA$",sep=""))
      if (length(data_name)==0) { #lowercase extension
        data_name=list.files(path="unzipped",pattern=paste0("^",ctr,".*?",ver,".*?\\.dta$",sep=""))
      }
      if (length(data_name)==0) { #lowercase countryname
        data_name=list.files(path="unzipped",pattern=paste0("^",tolower(ctr),".*?",ver,".*?\\.dta$",sep=""))
      }
      #name of the gps data
      shape_name=list.files(path="unzipped",pattern=paste0("^",ctr,".*?",ver,".*?\\.shp$",sep=""))
      
      #reading DHS data, keeping important variables,
      #and collapsing by cluster
      data=read.dta13(paste("unzipped",data_name,sep="/"),select.cols=c("hv001","hv271")) %>% mutate(hv271=hv271/100000) %>%
        collap( hv271  ~ hv001, fmean)
      #reading gps data
      gps= as.data.frame(st_read(paste("unzipped",shape_name,sep="/"),quiet=TRUE)) %>% subset(select=c("DHSID","DHSCLUST","LATNUM","LONGNUM"))
      #merging data and dropping cluster name (we keep cluster id)
      data=merge(data,gps,by.x="hv001",by.y="DHSCLUST") %>% select(-hv001)
      #appending dataset to our final dataset
      if (start==0) {
        full_dataset=data
        start=1
      } else {
        full_dataset=rbind(full_dataset,data)
      }
      }, #closing bracket at the beginning of trycatch
      error=function(cond) {
        print("something failed with this match")
      }
      ) #closing tryCatch
      #deleting unzipped files (we don't need them anymore)
      unlink("unzipped/*", recursive = T, force = T)
    }
    if (n_matches==0){
      print("no matches")
    }
  }
}


#mutating final dataset (extracting country and year)
full_dataset=full_dataset %>% mutate(ctr=substr(DHSID,1,2))
full_dataset=full_dataset %>% mutate(year=as.integer(substr(DHSID,3,6)))
full_dataset$ctr_name=countrycode(full_dataset$ctr,origin="dhs",destination="country.name")

write.csv(full_dataset,"processed/ground_truth.csv", row.names = FALSE)


full_dataset=full_dataset[full_dataset$year>2013,]
full_dataset=st_as_sf(full_dataset, coords = c("LONGNUM", "LATNUM"), 
             crs = 4326, agr = "constant")

africa_simple <- gisco_get_countries(region = "Africa", resolution = "20") %>% st_crop(xmin = -20, xmax = 60, ymin = -40, ymax = 40)

africa_simple$area=as.vector(st_area(africa_simple$geometry))/(10000000000)

graphics.off()
pdf(file="map_dhs.pdf")

ggplot(africa_simple) +
  geom_sf(fill=NA) +
  geom_sf_text(data=africa_simple[africa_simple$area>70,],aes(label = ISO3_CODE), size=2.5) +
  geom_sf(data=full_dataset$geometry,size=0.4,colour="red") +
  coord_sf(datum = NA) + 
  theme(panel.background = element_rect(fill = NA), plot.title = element_text(hjust = 0.4),
        axis.title.x=element_blank(),axis.title.y=element_blank())

dev.off()