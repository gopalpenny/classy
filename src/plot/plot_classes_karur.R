# plot_classes_karur

library(raster)
library(tidyverse)

out_path <- ggp::fig_set_output("plot_classes_karur")


classy_path <- "/Users/gopal/Google Drive/_Research/Research projects/ML/classy/classy_downloads"
karur_path <- file.path(classy_path,"karur_crops_2015_2021.tif")
r_karur <- stack(karur_path)

r_karur_change <- r_karur

for (i in 2:nlayers(r_karur)) {
  r_karur_change[[i]][r_karur[[1]] == r_karur[[i]]] <- NA
}


png(file.path(out_path, paste0("karur_crops_",6+2014,".png")), width = 1200, height = 800)
plot(r_karur[[6]])
dev.off()

for (i in 1:nlayers(r_karur)){
  png(file.path(out_path, paste0("karur_crops_change_",i+2014,".png")), width = 1200, height = 800)
  plot(r_karur_change[[i]])
  dev.off()
}

# names(r_karur_change) <- gsub
# plot(r_karur_change)
# RStoolbox::ggR(r_karur$X1_crops, geom_raster = TRUE) +
#   scale_fill_continuous()


karur_df <- as_tibble(as.data.frame(r_karur, xy = TRUE))

# karur_df %>% na.omit()

karur_df_count <- karur_df %>% 
  group_by(across(everything())) %>%
  summarize(n = n()) %>%
  arrange(desc(n)) %>%
  ungroup() %>%
  mutate(id=row_number()) %>%
  group_by(X0_crops) %>%
  arrange(X0_crops, desc(n)) %>%
  mutate(id_2015=row_number()) %>%
  filter(id_2015 <= 10)


karur_count <- karur_df_count %>% 
  ungroup() %>%
  mutate(across(starts_with("X"),
                function(x) factor(x, levels = 3:0, labels = c("Triple","Double","Single","Fallow"))),
         crop_2015 = X0_crops) %>%
  pivot_longer(cols = starts_with("X"), names_to = "varname", values_to = "crop") %>%
  group_by(id) %>%
  mutate(num_unique = length(unique(crop)),
         year = as.numeric(gsub("X([0-9]+)_.*","\\1",varname))+2015) %>%
  filter(num_unique > 1 | crop != "Fallow")


p_karur_alluvial <- ggplot(karur_count,
       aes(x = year, y = n * 900 / 1e4, stratum = crop, alluvium = id,
           fill = crop, label = crop)) +
  # scale_fill_brewer(type = "qual", palette = "Set2") +
  geom_flow(stat = "alluvium", lode.guidance = "frontback",
            color = "darkgray") +
  geom_stratum() +
  # geom_text(stat = "stratum", aes(label = after_stat(stratum)))
  scale_x_continuous(breaks = 2015:2021) +
  scale_y_continuous("Area (ha)",labels = scales::label_number(scale_cut = scales::cut_si("")))+
  scale_fill_manual('Cropping\nintensity',values = c("forestgreen","lightgreen","gold2","tan")) +
  ggp::t_manu() %+replace% theme(axis.title.x = element_blank())


ggsave("karur_alluvial.png", p_karur_alluvial, width = 5, height = 3, path = out_path)

ggp::obj_size(karur_df)
