# install.packages("osmdata")

# https://cran.r-project.org/web/packages/osmdata/vignettes/osmdata.html

library(osmdata)
library(sf)
library(ggplot2)

q <- opq(bbox = "Battambang, Cambodia") %>%
  add_osm_feature(key = c("waterway","natural"))


q <- 

btm_water <- opq(bbox = "Battambang, Cambodia") %>%
  add_osm_feature(key = "natural", value = "water") %>%
  osmdata_sf()

btm_waterways<- opq(bbox = "Battambang, Cambodia") %>%
  add_osm_feature(key = "waterway") %>%
  osmdata_sf()
# head(q$available_features())

ggp::obj_size(btm)

names(btm$osm_lines)
ggplot() + 
  geom_sf(data = btm_waterways$osm_lines, aes(color = waterway)) +
  geom_sf(data = btm_water$osm_polygons, aes(fill = water), color = NA)


ggplot() + 
  # geom_sf(data = btm$osm_lines, aes(color = waterway)) +
  # geom_sf(data = btm$osm_multilines, aes(color = waterway)) +
  geom_sf(data = btm$osm_polygons, aes(fill = water, color = water))


btm$osm_polygons #%>% dplyr::select(-geometry)
anames
