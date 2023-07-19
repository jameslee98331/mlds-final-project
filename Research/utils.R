create_sim_df <- function(data, true_labels) {
  # Helper function to create dataframes for simulation from
  # generated data
  df = as.data.frame(cbind(data, true_labels))
  df$true_labels = as.factor(df$true_labels)
  colnames(df) = c("x", "y", "true_label")
  return(df)
}

plot_generated_data = function(df, name) {
  df %>%
    ggplot() +
    geom_point(mapping = aes(x = x, y = y, color = true_label)) +
    ggtitle(glue("Simulated data - Mixture of {K_0} {name}"))
}
