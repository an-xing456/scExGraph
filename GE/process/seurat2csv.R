library(hdf5r)
library(Seurat)
library(readr)
library(dplyr)
library(hdf5r)
library(Matrix)

# Function to read RDS file and create Seurat object
read_and_create_seurat <- function(rds_file_path) {
  seurat_object <- readRDS(rds_file_path)
  return(seurat_object)
}

# Function to convert Seurat data to sparse matrix (Matrix Market format)
convert_seurat_to_sparse <- function(seurat_object, output_file_path) {
  data_matrix <- GetAssayData(seurat_object, slot = "data")
  sparse_matrix <- as(data_matrix, "dgCMatrix")
  writeMM(sparse_matrix, file = output_file_path)
}

# Function to filter by cell type intersection
filter_by_celltype_intersection <- function(seurat_object, intersection, mapping_vector, save_path) {
  filtered_cells <- which(seurat_object@meta.data$`Celltype..major.lineage.` %in% intersection)
  seurat_object <- subset(seurat_object, cells = filtered_cells)
  seurat_object@meta.data$encoded_celltypes <- mapping_vector[seurat_object@meta.data$`Celltype..major.lineage.`]
  
  write.table(filtered_cells, file = save_path, sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
  return(seurat_object)    
}


main <- function(train_rds_path, test_rds_path, test_pred_label, test_pred_index, process_path) {
  pbmc_train <- read_and_create_seurat(train_rds_path)
  pbmc_test <- read_and_create_seurat(test_rds_path)

  common_genes <- intersect(rownames(pbmc_train), rownames(pbmc_test))
  pbmc_train <- subset(pbmc_train, features = common_genes)
  pbmc_test <- subset(pbmc_test, features = common_genes)
  
  # Replace with predicted values
  indices <- as.integer(read.table(test_pred_index, header = FALSE)$V1)
  pred_labels <- read.csv(test_pred_label)
  
  filtered_cells <- rownames(pbmc_test@meta.data)[indices]
  pbmc_test <- subset(pbmc_test, cells = filtered_cells)
  pbmc_test@meta.data$`Celltype..major.lineage.` <- as.character(pred_labels$y_pred_labels)
  
  # Filter by cell type intersection and save mapping
  train_filtered_indices_path <- paste0(process_path, train_name, "_index.txt")
  test_filtered_indices_path <- paste0(process_path, test_name, "_index.txt")
  cell_types_train <- unique(pbmc_train@meta.data$`Celltype..major.lineage.`)
  cell_types_test <- unique(pbmc_test@meta.data$`Celltype..major.lineage.`)
  intersection <- intersect(cell_types_train, cell_types_test)
  
  cell_type_mapping <- unique(data.frame(Cell_type = intersection, encoded_celltypes = 0:(length(intersection)-1)))
  print(cell_type_mapping)
  mapping_vector <- setNames(cell_type_mapping$encoded_celltypes, cell_type_mapping$Cell_type)
  pbmc_train <- filter_by_celltype_intersection(pbmc_train, intersection, mapping_vector, train_filtered_indices_path)
  intersection1 <- unique(c(intersection, "Malignant"))
  pbmc_test <- filter_by_celltype_intersection(pbmc_test, intersection1, mapping_vector, test_filtered_indices_path)
  
  mapping_file_path <- paste0(process_path, "mapping.csv")
  write.csv(cell_type_mapping, file = mapping_file_path, row.names = FALSE, quote = FALSE)
  
  # Save gene names
  gene_names_train <- rownames(pbmc_train)
  gene_names_test <- rownames(pbmc_test)
  
  gene_names_train_path <- paste0(process_path, train_name, "_gene_names.csv")
  gene_names_test_path <- paste0(process_path, test_name, "_gene_names.csv")
  write.csv(gene_names_train, file = gene_names_train_path, row.names = FALSE)
  write.csv(gene_names_test, file = gene_names_test_path, row.names = FALSE)
  
  meta_train <- pbmc_train@meta.data[, c("Cell", "Celltype..major.lineage.", "encoded_celltypes")]
  meta_test <- pbmc_test@meta.data[, c("Cell", "Celltype..major.lineage.", "encoded_celltypes")]
  
  meta_train_data_path <- paste0(process_path, train_name, "_meta_data.csv")
  meta_test_data_path <- paste0(process_path, test_name, "_meta_data.csv")
  write.csv(meta_train, file = meta_train_data_path, row.names = FALSE)
  write.csv(meta_test, file = meta_test_data_path, row.names = FALSE)
  
  # Convert and save sparse matrix in Matrix Market format
  train_sparse_path <- paste0(process_path, train_name, "_sparse_data.mtx")
  test_sparse_path <- paste0(process_path, test_name, "_sparse_data.mtx")
  
  convert_seurat_to_sparse(pbmc_train, train_sparse_path)
  convert_seurat_to_sparse(pbmc_test, test_sparse_path)
  
  cat("Sparse matrix, gene names, and meta data saved.\n")
}


args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop("Please provide five arguments: baset_path,pre_path,train_name,test_name and auxiliary_name")
}
base_path <- args[1]
pre_path <- args[2]
train_name <- args[3]
test_name <- args[4]
auxiliary_name <- args[5]

print(base_path)
print(pre_path)
print(train_name)
print(test_name)
print(auxiliary_name)

train_rds_path <- paste0(base_path, "normal/processed/", train_name, ".rds")
test_rds_path <- paste0(base_path, "cancer/processed/", test_name, ".rds")

test_num <- as.numeric(gsub(".*[^0-9]([0-9]+).*", "\\1", test_name))
auxiliary_num <- as.numeric(gsub(".*[^0-9]([0-9]+).*", "\\1", auxiliary_name))

if (auxiliary_num < test_num) {
  ordered_names <- paste0(auxiliary_name, "_", test_name)
} else {
  ordered_names <- paste0(test_name, "_", auxiliary_name)
}
test_pred_label <- paste0(pre_path, ordered_names, "/pred_labels/",test_name,".csv")
test_pred_index <- paste0(pre_path, ordered_names,"/",test_name,"_index.txt")

process_path <- paste0(base_path,"GE_process/", train_name, "_", test_name, "_", auxiliary_name, "/")
if (!dir.exists(process_path)) {
  dir.create(process_path, recursive = TRUE)
}

main(train_rds_path, test_rds_path, test_pred_label, test_pred_index, process_path)











