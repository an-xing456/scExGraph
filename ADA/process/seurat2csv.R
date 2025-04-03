library(Seurat)
library(readr)
library(dplyr)
library(patchwork)
library(ggplot2)

# Function to read RDS file and create Seurat object
read_and_create_seurat <- function(rds_file_path) {
  seurat_object <- readRDS(rds_file_path)
  return(seurat_object)
}

# Function to filter genes
filter_genes <- function(seurat_object, common_genes) {
  seurat_object <- subset(seurat_object, features = common_genes)
  return(seurat_object)
}

# Function to convert Seurat data 
convert_seurat_data <- function(seurat_object) {
  data_matrix <- GetAssayData(seurat_object, slot = "data")
  sparse_matrix <- as(data_matrix, "dgCMatrix")
  dense_matrix <- as.matrix(sparse_matrix)
  return(dense_matrix)
}

# Function to filter Seurat object by intersected cell types
filter_by_celltype_intersection <- function(seurat_object, intersection, mapping_vector, save_path) {
  if (!"Celltype..major.lineage." %in% colnames(seurat_object@meta.data)) {
    stop("Celltype..major.lineage. column is missing from the metadata.")
  }

  filtered_cells <- which(seurat_object@meta.data$`Celltype..major.lineage.` %in% intersection)
  seurat_object <- subset(seurat_object, cells = filtered_cells)
  seurat_object@meta.data$encoded_celltypes <- mapping_vector[seurat_object@meta.data$`Celltype..major.lineage.`]

  # Save filtered cell indices
  write.table(filtered_cells, file = save_path, sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
  return(seurat_object)
}

main <- function(train_rds_path, test_rds_path, process_path, train_name, test_name) {
  pbmc_train <- read_and_create_seurat(train_rds_path)
  pbmc_test <- read_and_create_seurat(test_rds_path)

  # Filter genes
  common_genes <- intersect(rownames(pbmc_train), rownames(pbmc_test))
  pbmc_train <- filter_genes(pbmc_train, common_genes)
  pbmc_test <- filter_genes(pbmc_test, common_genes)

  # Filter by cell type intersection and save mapping
  train_filtered_indices_path <- file.path(process_path, paste0(train_name, "_index.txt"))
  test_filtered_indices_path <- file.path(process_path, paste0(test_name, "_index.txt"))
  cell_types_train <- unique(pbmc_train@meta.data$`Celltype..major.lineage.`)
  cell_types_test <- unique(pbmc_test@meta.data$`Celltype..major.lineage.`)
  intersection <- intersect(cell_types_train, cell_types_test)
  cell_type_mapping <- unique(data.frame(Cell_type = intersection, encoded_celltypes = 0:(length(intersection)-1)))
  mapping_vector <- setNames(cell_type_mapping$encoded_celltypes, cell_type_mapping$Cell_type)

  pbmc_train <- filter_by_celltype_intersection(pbmc_train, intersection, mapping_vector, train_filtered_indices_path)
  pbmc_test <- filter_by_celltype_intersection(pbmc_test, intersection, mapping_vector, test_filtered_indices_path)

  mapping_file_path <- file.path(process_path, "mapping.csv")
  write.csv(cell_type_mapping, file = mapping_file_path, row.names = FALSE, quote = FALSE)

  # Merge datasets
  pbmc_train$orig.ident <- "train"
  pbmc_test$orig.ident <- "test"
  combined <- merge(pbmc_train, pbmc_test)
  Idents(combined) <- combined$orig.ident

  # Identify variable genes and save names
  combined <- FindVariableFeatures(combined, selection.method = "vst", nfeatures = 2000)
  variable_genes <- VariableFeatures(combined)
  variablefile_path <- file.path(process_path, "variable_genes.csv")
  write.csv(variable_genes, file = variablefile_path, row.names = FALSE)
  cat("2000 variable genes saved as CSV.\n")

  # Split combined into train and test datasets
  train_data <- subset(combined, subset = orig.ident == "train", features = variable_genes)
  test_data <- subset(combined, subset = orig.ident == "test", features = variable_genes)

  # save meta.data
  norm_train_data <- convert_seurat_data(train_data)
  norm_test_data <- convert_seurat_data(test_data)
  norm_train_data_path <- file.path(process_path, paste0(train_name, "_norm_data.csv"))
  norm_test_data_path <- file.path(process_path, paste0(test_name, "_norm_data.csv"))
  write.csv(norm_train_data, file = norm_train_data_path, row.names = TRUE)
  write.csv(norm_test_data, file = norm_test_data_path, row.names = TRUE)

  meta_train <- train_data@meta.data[, c("Cell", "Celltype..major.lineage.", "encoded_celltypes")]
  meta_test <- test_data@meta.data[, c("Cell", "Celltype..major.lineage.", "encoded_celltypes")]
  meta_train_data_path <- file.path(process_path, paste0(train_name, "_meta_data.csv"))
  meta_test_data_path <- file.path(process_path, paste0(test_name, "_meta_data.csv"))
  write.csv(meta_train, file = meta_train_data_path, row.names = FALSE)
  write.csv(meta_test, file = meta_test_data_path, row.names = FALSE)
  cat("data and meta data saved as CSV.\n")
}

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Please provide three arguments: base_path, train_name, and test_name")
}
base_path <- args[1]
train_name <- args[2]
test_name <- args[3]

print(base_path)
print(train_name)
print(test_name)

train_rds_path <- file.path(base_path, "cancer", "processed", paste0(train_name, ".rds"))
test_rds_path <- file.path(base_path, "cancer", "processed", paste0(test_name, ".rds"))

train_num <- as.numeric(gsub(".*[^0-9]([0-9]+).*", "\\1", train_name))
test_num <- as.numeric(gsub(".*[^0-9]([0-9]+).*", "\\1", test_name))
if (train_num < test_num) {
  ordered_names <- paste0(train_name, "_", test_name)
} else {
  ordered_names <- paste0(test_name, "_", train_name)
}

process_path <- file.path(base_path, "ADA_process", ordered_names)
if (!dir.exists(process_path)) {
  dir.create(process_path, recursive = TRUE)
}

# ADA
main(train_rds_path, test_rds_path, process_path, train_name, test_name)
