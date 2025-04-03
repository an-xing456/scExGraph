library(Seurat)
library(dplyr)
library(readr)

# Function: Read HDF5 file and create Seurat object
read_and_create_seurat <- function(file_path, label_path, project_name) {
  data_sample <- Read10X_h5(file_path)
  
  meta_data <- read_tsv(label_path) %>%
    as.data.frame() %>%
    mutate_all(as.character) %>%
    rename_with(make.names) %>%
    select(Cell, Celltype..major.lineage., Patient)
  rownames(meta_data) <- meta_data$Cell
  
  seurat_object <- CreateSeuratObject(counts = data_sample, project = project_name)
  seurat_object <- AddMetaData(seurat_object, meta_data)
  
  return(seurat_object)
}

# Set paths
# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Please provide two arguments: base_path and name")
}
base_path <- args[1]
name <- args[2]

print(base_path)
print(name)
process_path <-paste0(base_path,"processed/")
file_path <- paste0(base_path, name, "_expression.h5")
label <- paste0(base_path, name, "_CellMetainfo_table.tsv")

# Create directory if it doesn't exist
if (!dir.exists(process_path)) {
  dir.create(process_path, recursive = TRUE)
}

# Read data and create Seurat object
pbmc <- read_and_create_seurat(file_path, label, "train")

# Initialize list to store summary statistics
cell_type_summary <- list()

# Calculate statistics for the complete dataset
complete_cell_type_count <- table(pbmc$Celltype..major.lineage.)
complete_health_cells <- sum(complete_cell_type_count[names(complete_cell_type_count) != "Malignant"])
complete_total_cells <- sum(complete_cell_type_count)

# Save statistics of the complete dataset to the list
cell_type_summary[[name]] <- c(complete_cell_type_count, "Total Cells" = complete_total_cells, "Health Cells" = complete_health_cells)

# Save the complete Seurat object
saveRDS(pbmc, file = paste0(process_path, name, ".rds"))

name <- gsub("(inDrop|10X|aPD1|aPD1aCTLA4)", "", name)
name <- gsub("_+", "_", name)
name <- gsub("^_|_$", "", name)

# Get unique values of the Patient column
patients <- unique(pbmc$Patient)

# Iterate over each Patient and save subset statistics
for (i in seq_along(patients)) {
  # Get current Patient name
  patient_name <- patients[i]

  # Generate subset name without zero-padding
  subset_name <- paste0(name, sprintf("%02d", i - 1))

  # Extract subset based on current Patient
  subset_cells <- WhichCells(pbmc, expression = Patient == patient_name)
  seurat_subset <- subset(pbmc, cells = subset_cells)

  # Ensure consistent factor levels for cell types
  all_cell_types <- levels(factor(pbmc$Celltype..major.lineage.))
  cell_type_count <- table(factor(seurat_subset$Celltype..major.lineage., levels = all_cell_types))

  # Calculate the number of healthy cells (assuming "Malignant" is the unhealthy cell type)
  health_cells <- sum(cell_type_count[names(cell_type_count) != "Malignant"])

  # Calculate the total number of cells
  total_cells <- sum(cell_type_count)

  # Save subset statistics to the list
  cell_type_summary[[subset_name]] <- c(cell_type_count, "Total Cells" = total_cells, "Health Cells" = health_cells)

  # Save Seurat object for each subset
  saveRDS(seurat_subset, file = paste0(process_path, subset_name, ".rds"))
}


# Convert the list of statistics to a data frame
summary_df <- as.data.frame(do.call(cbind, cell_type_summary))

# Define the path to save the summary
file_path <- paste0(process_path, name, "_summary.csv")

# Check if the file exists
if (file.exists(file_path)) {
  write.csv(summary_df, file = file_path, row.names = TRUE)
  
} else {
  # If the file doesn't exist, create a new file and save the data
  write.csv(summary_df, file = file_path, row.names = TRUE)
}

print("------------------------------complete data split-------------------------------")
