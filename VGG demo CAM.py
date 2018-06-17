from keras.applications import VGG16
import keras.backend as K
import numpy as np
import cv2
import os
from keras.applications.vgg16 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Lambda, Dense

# from keras.applications.vgg16 import preprocess_input, decode_predictions
# from keras.models import Model

# sys.path.append('E:\\Users\\Ross\\Downloads\\Python\\(Top)常用函數&方法\\Keras\\CAM')

os.chdir("E:\\\\Users\\\\Ross\\Downloads\\Python\\(Top)常用函數&方法\\Dataset")

# 讀入圖片
img_path = 'tiger.jpg'

tiger = cv2.imread(img_path)
tiger_origin = cv2.imread(img_path)
tiger_origin = cv2.resize(tiger_origin, (224, 224))
tiger = cv2.resize(tiger, (224, 224))
# 要丟進取特徵需要resize成模型可吃的形狀
tiger = tiger.reshape(1, 224, 224, 3)

tiger = np.array(tiger, dtype=np.float64)
tiger = preprocess_input(tiger)

# GAP
def global_average_pooling(x):
    return K.mean(x, axis=(1, 2))  # 要對應長寬

# 載入VGG模型
model = VGG16(weights='imagenet', include_top=True)
model.summary()
# 接上自訂GAP層
GAP = Lambda(global_average_pooling)(model.layers[-5].output)
Out = Dense(10, activation='softmax')(GAP)
vgg_conv = Model(model.input, Out)
vgg_conv.summary()

# 抓出最後一層的GAP連softmax權重，共有Feature Map數量*最後一層Nodes個。
class_weights = vgg_conv.layers[-1].get_weights()[0]  # [0]weights & [1]bias
class_weights.shape  # 512個feature map，每個有7個權重

# temp_img = cv2.resize(tiger, (224, 224))
# img = temp_img.reshape(1, 224, 224, 3)
# img.shape

# 定義最後一層conv層
vgg_conv.layers[-3].name = "conv_final"
layer_dict = dict([(layer.name, layer) for layer in model.layers])
final_conv_layer = layer_dict['conv_final']

# 抓取原圖size，得到heatmap後就可以resize回去
width, height, _ = tiger_origin.shape

# final_conv_layer = get_output_layer(model, "conv_final")  # 取出最後一層CONV的Feature Map
get_output = K.function([vgg_conv.layers[0].input], [final_conv_layer.output, vgg_conv.layers[-1].output])  # 將FM和預測類別取出
[conv_outputs, predictions] = get_output([tiger])  # 丟入IMG去得到FM和預測類別 (1, 7, 7, 512)
conv_outputs = conv_outputs[0, :, :, :]  # (1, 7, 7, 512) -> (7, 7, 512)

# 將GAP連接預測值node的權重乘上最後一層conv的output Feature Map
cam = np.zeros(shape=conv_outputs.shape[0:2], dtype=np.float32)
for i, w in enumerate(class_weights[:, np.argmax(predictions)]):
    cam += w * conv_outputs[:, :, i]
cam -= np.min(cam)
cam /= np.max(cam)
cam = cv2.resize(cam, (height, width))
heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
heatmap[np.where(cam < 0.2)] = 0

# 修正原圖色彩
tiger_origin = cv2.cv2.cvtColor(tiger_origin, cv2.COLOR_RGB2BGR)
# heatmap與原圖疊起來
img = (heatmap * .5 + tiger_origin * .5).astype('uint8')
print(img.dtype)
img = img.reshape(224, 224, 3)
plt.imshow(img)
plt.show()
plt.imshow(tiger_origin)
plt.show()

# Load_VGG_model的函數
def load_VGG_model():
    model = VGG16(weights='imagenet', include_top=True)
    GAP = Lambda(global_average_pooling)(model.layers[-5].output)
    Out = Dense(10, activation='softmax')(GAP)

    vgg_conv = Model(model.input, Out)

    return vgg_conv


vgg_conv = load_VGG_model()

# 視覺化CAM的函數
def Visualize_Cam(image_path, cute_model, last_conv_pos, W=299, H=299, ch=3):
    # 必須為GAP接SOFTMAX的model
    origin_image = cv2.imread(image_path)
    width, height, _ = origin_image.shape
    # 將圖整理成可預測的形狀
    pred_image = cv2.resize(origin_image, (W, H))
    print(pred_image.shape)
    pred_image = np.expand_dims(pred_image, axis=0)
    print(pred_image.shape)
    # 首先取出GAP連SOFTMAX的權重
    class_weights = cute_model.layers[-1].get_weights()[0]
    # 定義最後一層conv的名稱為final_conv
    cute_model.layers[last_conv_pos].name = "final_conv"
    # 產生每個layer名稱的字典
    layer_dict = dict([(layer.name, layer) for layer in cute_model.layers])
    # 抓出最後一層conv_layer
    final_conv_layer = layer_dict['final_conv']
    get_output = K.function([cute_model.layers[0].input],
                            [final_conv_layer.output, cute_model.layers[-1].output])  # 將FM和預測類別取出
    [conv_outputs, predictions] = get_output([pred_image])  # 丟入IMG去得到FM和預測類別 (1, 7, 7, 512)
    conv_outputs = conv_outputs[0, :, :, :]  # 去掉batch的維度
    # 取出長寬，製作空白矩陣準備線性相加

    cam = np.zeros(shape=conv_outputs.shape[0:2], dtype=np.float32)
    for i, w in enumerate(class_weights[:, np.argmax(predictions)]):
        cam += w * conv_outputs[:, :, i]
    cam -= np.min(cam);
    cam /= np.max(cam)
    cam = cv2.resize(cam, (height, width))  # cv2的長寬是反過來的
    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0
    # 修正原圖色彩
    origin_image = cv2.cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR)
    # 將cam疊上原圖
    pred_image = (heatmap * .5 + origin_image * 0.5).astype('uint8')
    # pred_image = np.clip(pred_image, 0, 255)
    # print(img.dtype)
    # pred_image = pred_image.reshape(W, H, ch)
    plt.imshow(pred_image)
    plt.show()

def model_pred(cute_model=None, img_matrix=None, W=299, H=299, ch=3):
    illness_dict = {'0': "Melanoma", "1": "Seborrheic Keratosis", "2": "Normal"}
    # img_matrix的維度要是(1, W, H, ch)
    img_matrix = img_matrix.reshape(1, W, H, ch)
    result = cute_model.predict(img_matrix)
    illness = illness_dict[str(np.argmax(result))]

    return illness

os.chdir("E:\\\\Users\\\\Ross\\Downloads\\Python\\(Top)常用函數&方法\\Dataset")
# os.chdir("E:\\Users\\Ross\\Downloads\\Python\\INFORMS競賽\\informs hackathon\\ISIC-2017_Training_Data")
Visualize_Cam("dog2.jpg", vgg_conv, -3, W=224, H=224)
# Visualize_Cam("ISIC_0000006.jpg", model2, -3, W=299, H=299)
