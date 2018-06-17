# reticulate::use_python("C:\\Users\\Ross\\Anaconda3\\envs\\tensorflowgpu\\python.exe",required = T)
library("reticulate")
# use_condaenv(condaenv = "tensorflowgpu")
# 改變python環境
py_config()
# py_discover_config(use_environment = "tensorflowgpu")
# load套件
py_available(initialize = T)
os <- import("os")
keras <- import("keras")
cv2 <- import('cv2', convert = FALSE)
np <- import("numpy")

setwd("E:\\Users\\Ross\\Downloads\\Python\\INFORMS競賽\\informs hackathon")
# 載入模組
module <- import_from_path("hackathon_module","E:\\Users\\Ross\\Downloads\\Python\\INFORMS競賽\\informs hackathon")
# model_VGG <- module$load_VGG_model()

model_lin <- keras$models$load_model("model_hack.h5")
model_chen <- keras$models$load_model("hackthon_73.h5")
# model_chen$summary()
# 畫CAM圖(採用model_chen for 畫圖)
module$Visualize_Cam(image_path = "ISIC-2017_Training_Data/ISIC_0000007.jpg",cute_model = model_chen,
                     last_conv_pos = -3L, W =  299L, H = 299L)

getwd()
# 載入圖片並預測(採用model_lin for 預測)
img <-cv2$imread('ISIC-2017_Training_Data/ISIC_0000001.jpg')
result = module$model_pred(cute_model = model_lin, img_matrix = img, W = 299L, H = 299L)
result

# 秀圖片
library("jpeg")
jj <- readJPEG("output_img.jpg",native=TRUE)
plot(0:1,0:1,type="n",ann=FALSE,axes=FALSE)
rasterImage(jj,0,0,1,1)

