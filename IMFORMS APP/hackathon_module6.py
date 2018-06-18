import numpy as np
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import cv2
import keras.backend as K
from keras.models import Model
from keras.layers import Lambda, Dense


def global_average_pooling(x):
    return K.mean(x, axis=(1, 2))  # 要對應長寬



def Visualize_Cam(origin_image, cute_model, last_conv_pos, W=299, H=299, ch=3, img_no=None):
    # 必須為GAP接SOFTMAX的model
    # origin_image = cv2.imread(image_path)
    width, height, _ = origin_image.shape
    # 將圖整理成可預測的形狀
    pred_image = cv2.resize(origin_image, (W, H))
    print(pred_image.shape)
    pred_image = np.expand_dims(pred_image, axis=int(0))
    print(pred_image.shape)
    # 首先取出GAP連SOFTMAX的權重
    class_weights = cute_model.layers[-int(1)].get_weights()[int(0)]
    # 定義最後一層conv的名稱為final_conv
    cute_model.layers[last_conv_pos].name = "final_conv"
    # 產生每個layer名稱的字典
    layer_dict = dict([(layer.name, layer) for layer in cute_model.layers])
    # 抓出最後一層conv_layer
    final_conv_layer = layer_dict['final_conv']
    get_output = K.function([cute_model.layers[int(0)].input],
                            [final_conv_layer.output, cute_model.layers[-int(1)].output])  # 將FM和預測類別取出
    [conv_outputs, predictions] = get_output([pred_image])  # 丟入IMG去得到FM和預測類別 (1, 7, 7, 512)
    conv_outputs = conv_outputs[int(0), :, :, :]  # 去掉batch的維度
    # 取出長寬，製作空白矩陣準備線性相加
    cam = np.zeros(shape=conv_outputs.shape[int(0):int(2)], dtype=np.float32)
    for i, w in enumerate(class_weights[:, np.argmax(predictions)]):
        cam += w * conv_outputs[:, :, int(i)]
    cam -= np.min(cam);
    cam /= np.max(cam)
    cam = cv2.resize(cam, (height, width))  # cv2的長寬是反過來的
    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = int(0)
    # 修正原圖色彩
    origin_image = cv2.cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR)
    # 將cam疊上原圖
    pred_image = (heatmap * .5 + origin_image * 0.5).astype('uint8')
    # print(img.dtype)
    # pred_image = pred_image.reshape(W, H, ch)
    cv2.imwrite("predimg/output_img"+str(int(img_no))+".jpg", pred_image)


def hack_model():
    model = InceptionResNetV2(include_top=True, weights='imagenet')
    model.layers.pop()
    new_layer = Dense(int(3), activation='softmax', name='my_dense')

    inp = model.input
    out = new_layer(model.layers[-int(1)].output)

    model2 = Model(inp, out)

    return model2


def model_pred(cute_model=None, img_matrix=None, W=299, H=299, ch=3):
    # img_matrix = cv2.imread(image_path)
    illness_dict = {'0': "Melanoma", "1": "Seborrheic Keratosis", "2": "Normal"}
    # img_matrix的維度要是(1, W, H, ch)
    img_matrix = cv2.resize(img_matrix, (H, W))
    img_matrix = np.expand_dims(img_matrix, axis=int(0))
    result = cute_model.predict(img_matrix)
    illness = illness_dict[str(np.argmax(result))]

    return illness
