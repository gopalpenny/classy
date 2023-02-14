# plot classapp_ts.R

library(tidyverse)
thailand_path <- "/Users/gopal/Google Drive/_Research/Research projects/ML/classapp/app_data/Thailand"
classy_path <- "/Users/gopal/Google Drive/_Research/Research projects/ML/classy"
# out_path <- file.path(classy_path, "plot_thailand_ts")
# dir.create(out_path)
ts_path <- file.path(thailand_path, "Thailand_download_timeseries")
class_path <- file.path(thailand_path, "Thailand_classification/location_classification.csv")
out_path <- ggp::fig_set_output("plot_thailand_ts")


classes <- read_csv(class_path)
ts_files <- tibble(filename = list.files(ts_path))%>% 
  filter(!grepl("ts_status",filename)) %>%
  mutate(loc_id_char = gsub(".*loc([0-9]+)_.*","\\1", filename),
         loc_id = as.numeric(loc_id_char),
         satellite = gsub(".*loc[0-9]+_(.*)\\.csv*","\\1", filename)) %>%
  left_join(classes %>% dplyr::select(loc_id, class2019=Subclass2019), by = "loc_id")
ts_files


# plot_s2_loc_id <- function(filename) {
#   
# }

read_ts <- function(filename) {
  loc_id <- gsub(".*loc([0-9]+)_.*","\\1", filename)
  ts <- read_csv(file.path(ts_path, filename))%>%
    mutate(ndvi = (B8 - B4) / (B8 + B4),
           date = lubridate::ymd(substr(image_id, 1, 8))) %>%
    dplyr::select(-image_id)
  ts$loc_id <- as.numeric(loc_id)
  return(ts)
}


filenames_single <- ts_files %>% filter(grepl("ingle", class2019), satellite == "s2") %>%
  slice(1:20) %>% pull(filename)
single <- do.call(bind_rows, lapply(filenames_single, read_ts))


filenames_double <- ts_files %>% filter(grepl("ouble", class2019), satellite == "s2") %>%
  slice(1:20) %>% pull(filename)
double <- do.call(bind_rows, lapply(filenames_double, read_ts))


filenames_plantation <- ts_files %>% filter(grepl("lantation", class2019), satellite == "s2") %>%
  slice(1:20) %>% pull(filename)
plantation <- do.call(bind_rows, lapply(filenames_plantation, read_ts))



p_single <- ggplot(single, aes(date, ndvi)) + 
  geom_smooth(span = 0.07) +
  geom_point(alpha = 0.25, size = 0.5) +
  facet_wrap(~loc_id) + ggtitle("Single")
ggsave("ts_single.png", p_single, path = out_path, width = 10, height = 8)
p_double <- ggplot(double, aes(date, ndvi)) + 
  geom_smooth(span = 0.07) +
  geom_point(alpha = 0.25, size = 0.5) +
  coord_cartesian(ylim = c(-0.5,1.5)) +
  facet_wrap(~loc_id) + ggtitle("Double")
ggsave("ts_double.png", p_double, path = out_path, width = 10, height = 8)
p_plantation <- ggplot(plantation, aes(date, ndvi)) + 
  geom_smooth(span = 0.07) +
  geom_point(alpha = 0.25, size = 0.5) +
  coord_cartesian(ylim = c(-0.5,1.5)) +
  facet_wrap(~loc_id) + ggtitle("Plantation")
ggsave("ts_plantation.png", p_plantation, path = out_path, width = 10, height = 8)


loc_ids_plot <- c(104, 131, 134)


filenames_examples <- ts_files %>% filter(loc_id %in% loc_ids_plot, satellite == "s2") %>%
  slice(1:20) %>% pull(filename)
loc_examples <- do.call(bind_rows, lapply(filenames_examples, read_ts))
loc_examples <- loc_examples %>% #group_by(loc_id) %>%
  mutate(num = as.numeric(as.factor(loc_id)),
         ndvi_adj = ndvi + num * 0.75)

year_divs = tibble(year = 2019:2022,
                   date = as.Date(paste0(year, "-05-01")))

p_loc_examples <- ggplot(loc_examples, aes(date, ndvi_adj, color = as.factor(loc_id))) + 
  geom_smooth(span = 0.07, se = FALSE) +
  geom_point(alpha = 0.4, size = 1) +
  ylab("NDVI") +
  geom_vline(data = year_divs, aes(xintercept = date), alpha = 0.5, linetype = "dashed") +
  # coord_cartesian(ylim = c(-0.2,1.2))# +
  # facet_wrap(~loc_id, ncol = 1) #+ #ggtitle("Plantation")
  ggp::t_manu() %+replace% theme(axis.title.x = element_blank(),
                                 axis.ticks.y = element_blank(),
                                 axis.text.y = element_blank(),
                                 legend.position = "none")
p_loc_examples
ggsave("ndvi_examples.png", p_loc_examples, path = out_path, width = 6, height = 3)

