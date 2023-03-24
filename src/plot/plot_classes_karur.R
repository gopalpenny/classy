# plot_classes_karur

library(raster)
library(tidyverse)
library(ggalluvial)
library(data.table)

out_path <- ggp::fig_set_output("plot_classes_karur")


classy_paths <- c("/Users/gopal/Google Drive/_Research/Research projects/ML/classy/classy_downloads",
                  "/Users/gopalpenny/Library/CloudStorage/GoogleDrive-gopalpenny@gmail.com/My Drive/_Research/Research projects/ML/classy/classy_downloads")
classy_path <- classy_paths[file.exists(classy_paths)][1]
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


karur_df <- as_tibble(as.data.frame(r_karur))


ggplot(karur_dt) + geom_raster(aes(x,y, fill= plot_highlight))

karur_df_count_wide <- karur_df %>% 
  # dplyr::select(-y,-y) %>%
  group_by(across(everything())) %>%
  summarize(n = n()) %>%
  arrange(desc(n)) %>%
  ungroup() %>%
  mutate(id=row_number()) %>%
  rowwise() %>%
  mutate(crop_order = paste(c_across(starts_with("X")), collapse = "_")) %>%
  group_by(X0_crops) %>%
  arrange(X0_crops, desc(n)) %>%
  mutate(id_2015=row_number()) %>%
  filter(id_2015 <= 10)

karur_count <- karur_df_count_wide %>% 
  ungroup() %>%
  mutate(across(starts_with("X"),
                function(x) factor(x, levels = 3:0))) %>% #, labels = c("Triple","Double","Single","Fallow")))) %>%
  mutate(plot_highlight = dplyr::case_when(
           grepl("3_3_3", crop_order) ~ "-3-3-3-",
           grepl("2", crop_order) ~ "-2-",
           grepl("1_1_1", crop_order) ~ "-1-1-1-",
           grepl("0_0_0_0", crop_order) ~ "-0-0-0-0-",
           TRUE ~ "-x-"
         ),
         plot_highlight = factor(plot_highlight, levels = c("-3-3-3-", "-2-","-1-1-1-","-0-0-0-0-","-x-")),
         crop_2015 = X0_crops) %>%
  pivot_longer(cols = starts_with("X"), names_to = "varname", values_to = "crop") %>%
  group_by(id) %>%
  mutate(num_unique = length(unique(crop)),
         year = as.numeric(gsub("X([0-9]+)_.*","\\1",varname))+2015) %>%
  filter(num_unique > 1 | crop != "Fallow")

# grepl("0_0_0_0", karur_count$crop_order)


p_karur_alluvial <- ggplot(karur_count,
       aes(x = year, y = n * 900 / 1e4, stratum = crop, alluvium = id,
           label = crop)) +
  scale_fill_manual("Select\nmulti-year\npatterns",values = c("forestgreen","lightgreen","gold2","tan","gray"), 
                    guide = guide_legend(order = 2)) +
  geom_flow(stat = "alluvium", lode.guidance = "frontback",
            color = "darkgray", aes(fill = plot_highlight)) +
  ggnewscale::new_scale_fill() +
  geom_stratum(aes(fill = crop)) +
  scale_fill_manual('Annual\ncropping\nintensity',values = c("forestgreen","lightgreen","gold2","tan"), 
                    guide = guide_legend(order = 1)) +
  # geom_text(stat = "stratum", aes(label = after_stat(stratum)))
  scale_x_continuous(breaks = 2015:2021) +
  scale_y_continuous("Area (ha)",labels = scales::label_number(scale_cut = scales::cut_si("")))+
  ggp::t_manu() %+replace% 
  theme(axis.title.x = element_blank(),legend.box = "horizontal",
        axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 10),
        axis.title.y = element_text(size = 10),
        legend.title = element_text(size = 10))
p_karur_alluvial

ggsave("karur_alluvial.png", p_karur_alluvial, width = 5, height = 2.25, path = out_path)

ggp::obj_size(karur_df)


# Map  patterns of change
karur_dt <- data.table(karur_df)
karur_dt[, crop_order := paste(X0_crops,X1_crops,X2_crops,X3_crops,X4_crops,X5_crops,X6_crops, sep = "_")]

karur_dt[, plot_highlight := as.numeric(0)]
# karur_dt$plot_highlight <- as.numeric(0)
karur_dt[grepl("3_3_3",crop_order), plot_highlight:=4]
karur_dt[crop_order == "1_0_0_0_1_2_1", plot_highlight:=3]
karur_dt[crop_order == "1_0_0_0_1_1_1", plot_highlight:=2]
karur_dt[crop_order == "0_0_0_0_0_1_1", plot_highlight:=1]

r_karur_patterns <- r_karur[[1]]
values(r_karur_patterns) <- karur_dt$plot_highlight
writeRaster(r_karur_patterns, file.path(out_path, "karur_crop_patterns.tif"))
plot(r_karur_patterns)



# Map  patterns of change
karur_dt2 <- data.table(karur_df)
karur_dt2[, crop_order := paste(X0_crops,X1_crops,X2_crops,X3_crops,X4_crops,X5_crops,X6_crops, sep = "_")]

karur_dt2[, plot_highlight := as.numeric(0)]
karur_dt2[grepl("3_3_3",crop_order), plot_highlight:=4]
karur_dt2[grepl("2",crop_order), plot_highlight:=3]
karur_dt2[grepl("1_1_1",crop_order), plot_highlight:=2]
karur_dt2[grepl("0_0_0_0",crop_order), plot_highlight:=1]

r_karur_patterns2 <- r_karur[[1]]
values(r_karur_patterns2) <- karur_dt2$plot_highlight
writeRaster(r_karur_patterns2, file.path(out_path, "karur_crop_patterns2.tif"))
plot(r_karur_patterns2)



RStoolbox::ggR(r_karur_patterns2,geom_raster = TRUE) +
  geom_sf()

