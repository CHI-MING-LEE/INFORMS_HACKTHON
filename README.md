# INFORMS_HACKTHON

Markdown語法參考: https://github.com/emn178/markdown

### 主要技術 ####
**1. 採用InceptionResNetv2**</br>
**2. CAM的寫法py檔**</br>
**3. Shiny呈現 -> Timer、reactiveValues**</br>
**4. R的reticulate套件Call Python使用，以及Shiny使用上的解決方法(例如虛擬路徑無法直接讀imread)**

* .h5檔因為太大load不上來

``R Shiny Timer``
```r
 predict_path<-reactiveValues(path=NULL) 
  observeEvent(reactiveTimer(1000)(),{ # Timer，每1000毫秒會更新一次這個block
      if(!file.exists("predimg/output_img.jpg")){
          predict_path$path=NULL
      }else{
          predict_path$path="predimg/output_img.jpg"
      }
  })
```

``R reticulate call python改環境``

```r
# 在library之前就要load環境才可以改
reticulate::use_python("C:\\Users\\Ross\\Anaconda3\\envs\\tensorflowgpu\\python.exe",required = T)
library("reticulate")
use_condaenv(condaenv = "tensorflowgpu")
# 改變python環境
py_config()
py_discover_config(use_environment = "tensorflowgpu")
# load套件
py_available(initialize = T)
```

``python CAM``

```python
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
    cam -= np.min(cam)
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
```

