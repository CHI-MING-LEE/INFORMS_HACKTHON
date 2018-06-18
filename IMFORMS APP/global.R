# Load the Python Module
library(reticulate)
if( !"os" %in% ls()){os <- import("os")}
if( !"keras" %in% ls()){keras <- import("keras")}
if( !"np" %in% ls()){np <- import("numpy")}
if( !"csv" %in% ls()){cv2 <- import('cv2', convert = FALSE)}
if( !"module" %in% ls()){module <- import_from_path("hackathon_module6",getwd())}


# load Lin, Chen model
if(!"model_lin" %in% ls()){model_lin <- keras$models$load_model("model_hack.h5")}
if(!"model_chen" %in% ls()){model_chen <- keras$models$load_model("hackthon_73.h5")}