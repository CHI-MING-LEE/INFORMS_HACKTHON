library(shiny)
library(ggplot2)
library(dplyr)
source("global.R")

font_size=4
function(input, output, session){
    
  #Introduction----
  output$toptext<-renderText({
    paste0(
      "<font size=\"6\"><b>The basic four techniques</b></font><br/>
      <font size=\"",font_size,"\">When you perform a physical assessment, you'll use four techniques:
      <font color=\"#090999\"><b>inspection</font></b>, 
      <font color=\"#090999\"><b>palpation</font></b>, 
      <font color=\"#090999\"><b>percussion</font></b>, and 
      <font color=\"#090999\"><b>auscultation</font></b>. Use them in sequence—unless you're performing an abdominal assessment. Palpation and percussion can alter bowel sounds, so you'd inspect, auscultate, percuss, then palpate an abdomen.</font>"
    )%>%HTML()
  })
  output$textimage<-renderImage({
    list(
      src = "img/bodyassessment.png"
      ,contentType = "image/png"
    )
  },deleteFile = F)
  output$text1<-renderText({
    paste0(
      "<font size=\"6\"><b>1. Inspection</b></font><br/>
<font size=\"",font_size,"\">Inspect each body system using vision, smell, and hearing to assess normal conditions and deviations. Assess for color, size, location, movement, texture, symmetry, odors, and sounds as you assess each body system.</font>"
    )%>%HTML()
  })
  output$text2<-renderText({
    paste0(
      "<font size=\"6\"><b>2. Palpation</b></font><br/>
<font size=\"",font_size,"\">Palpation requires you to touch the patient with different parts of your hands, using varying degrees of pressure. Because your hands are your tools, keep your fingernails short and your hands warm. Wear gloves when palpating mucous membranes or areas in contact with body fluids. Palpate tender areas last.</font>"
    )%>%HTML()
  })
  output$text3<-renderText({
    paste0(
      "<font size=\"6\"><b>3. Percussion</b></font><br/>
<font size=\"",font_size,"\">Percussion involves tapping your fingers or hands quickly and sharply against parts of the patient's body to help you locate organ borders, identify organ shape and position, and determine if an organ is solid or filled with fluid or gas.</font>"
    )%>%HTML()
  })
  output$text4<-renderText({
    paste0(
      "<font size=\"6\"><b>4. Auscultation</b></font><br/>
<font size=\"",font_size,"\">Auscultation involves listening for various lung, heart, and bowel sounds with a stethoscope.<br/>And we hope to use AI system help you to perform a physical assessment by yourself.</font>"
    )%>%HTML()
  })
  output$bottomtext<-renderText({
    paste0(
      "<font size=\"",font_size,"\">In our system, you can upload your image of specified site or voice, and the system will analysis your image and voice, then predict which class of diseases you may suffer or you are healthy right now. In the end we will give you some advice which can help you to do more detail inceptions or keep healthy for a long time.
We hope through AI help our world become more convenient and helpful no matter how you are and where you live.</font>"
    )%>%HTML()
  })
  #melanoma----
  img_file <- reactive({
    file <- input$upload_file
    if(is.null(file)){
      return(NULL)
    }
    else{
      file
    }
  })
  # illness photo
  output$upload_img <- renderUI({
    if(is.null(img_file())){
      htmlOutput("upload_img_False")
    }
    else{
      imageOutput("upload_img_True",width = 300,height = 240)
    }
  })
  output$upload_img_False<-renderText({
    HTML("<font color=\"#ff0000\"><font size=\"4\"><b>Image has not been upload yet</b></font>")
  })
  output$upload_img_True<-renderImage({
    list(
      src = img_file()$datapath
      ,contentType = "image/jpeg"
      ,width=300, height=240
    )
  })
  
  cv_img <- reactive({
      cv2$imread(
          paste0("tempimg/temp",user_img_no$num,".jpg")
      )
  })
  
  # predict & output CAM plot
  # 要有人呼叫到他，有dependency，按按鈕才有用 (原本是用eventReactive)
  observeEvent(input$python,{
      withProgress(message = 'Still Computing...',detail="part 1", value = 0, {
          # 預測
          result = module$model_pred(cute_model = model_lin, img_matrix = cv_img(), W = 299L, H = 299L)
          incProgress(0.5,detail = "part 1") # 當上面那行結束，進度條就達到50%
          # 畫圖
          module$Visualize_Cam(origin_image = cv_img(), cute_model = model_chen,
                               last_conv_pos = -3L, W =  299L, H = 299L, img_no = user_img_no$num)
          incProgress(1,detail = "part 2")
          Sys.sleep(0.2)
      })
      
      # 沒預測檔案就視同什麼結果都沒
      # 原本必須確保python_module()有被連動到，改成observeEvent就沒差
      output$Bigsuggestion <- renderText({
          if(is.null(predict_path$path)){
              return(NULL)
          }
          if(result=="Normal"){
              paste0("
                     <font color=\"#000000\"><font size=\"5\"><b>Congratulations!</b></font></font>
                     <font color=\"#000000\"><font size=\"5\"><b>You belong to ","<font color=\"#0da32d\">",result," group!</b></font></font></font><br/>"
              )%>%HTML()
          }
          else if(result=="Seborrheic Keratosis"){
              paste0("
                     <font color=\"#000000\"><font size=\"5\"><b>Don't worry!</b></font></font>
                     <font color=\"#000000\"><font size=\"5\"><b>You belong to ","<font color=\"#142f91\">",result," group!</b></font></font></font><br/>"
              )%>%HTML()
          }
          else{
              paste0("
                     <font color=\"#000000\"><font size=\"5\"><b>Warning!</b></font></font>
               <font color=\"#000000\"><font size=\"5\"><b>You belong to ","<font color=\"#e81717\">",result," group!</b></font></font></font><br/>"
              )%>%HTML()
          }
      })
      
      output$suggestion <- renderText({
          if(is.null(predict_path$path)){
              return(NULL)
          }
          if(result=="Normal"){
              paste0("
               <font size=\"",font_size,"\"> You are very healthy right now! Remember keeping a balanced diet and exercise three times a week</font>"
              )%>%HTML()
          }
          else if(result=="Seborrheic Keratosis"){
              paste0("
               <font size=\"",font_size,"\"> Seborrheic keratosis on human back. Multiple seborrheic keratoses on the dorsum of a patient with Leser–Trelat sign. Specialty Dermatology A seborrheic keratosis, also known as seborrheic verruca, basal cell papilloma, or a senile wart, is a non-cancerous (benign) skin tumour that originates from cells in the outer layer of the skin (keratinocytes), So don’t worry about it. Like liver spots, seborrheic keratoses are seen more often as people age.</font>"
              )%>%HTML()
          }
          else{
              paste0("
               <font size=\"",font_size,"\"> Using sunscreen and avoiding UV light may prevent Melanoma. Treatment is typically removal by surgery. In those with slightly larger cancers, nearby lymph nodes may be tested for spread. Most people are cured if spread has not occurred, so please take more careful diagnosis as soon as possible. For those in whom melanoma has spread, immunotherapy, biologic therapy, radiation therapy, or chemotherapy may improve survival.</font>"
              )%>%HTML()
          }
      })
  })
  
  # predicted imgage
  output$pred_img <- renderUI({
      if(is.null(img_file()$datapath)){
          htmlOutput("pred_img_False")
      }else if(is.null(predict_path$path)){
          htmlOutput("pred_img_False")
      }else{
          imageOutput("pred_img_True")
      }
  })
  output$pred_img_False<-renderText({
      HTML("<font color=\"#000000\"><font size=\"6\"><b>Magic:)</b></font>")
  })
  
  
  
  # timer update
  # 這邊也不一定要設path，只是個indicator說路徑裡有沒有檔案，因為預測輸出的路徑跟檔名都固定了
  predict_path<-reactiveValues(path=NULL) 
  observeEvent(reactiveTimer(1000)(),{ # Timer，每1000毫秒會更新一次這個block
      if(!file.exists(paste0("predimg/output_img",user_img_no$num,".jpg"))){
          predict_path$path=NULL
      }else{
          predict_path$path=paste0("predimg/output_img",user_img_no$num,".jpg")
      }
  })
  # 預測的檔案如果存在則輸出
  output$pred_img_True <- renderImage({
      if(is.null(predict_path$path)){
          return(list(
              src = "img/blank.jpg"
              ,contentType = "image/jpeg"
          ))
      }else{
          return(list(
              src = predict_path$path
              ,width=500
              ,height=400
              ,contentType = "image/jpeg"
          ))
      }
  },deleteFile = F)
  
  
  #----Button----
  observeEvent(input$launch, {
    confirmSweetAlert(
      session = session,
      inputId = "myconfirmation",
      type = "warning",
      title = "Want to confirm ?",
      danger_mode = TRUE
    )
  })
  observeEvent(input$myconfirmation,{
      if(input$myconfirmation){
          sendSweetAlert(
              session = session,
              title = "Success !!",
              text = "Thank you for your support!",
              type = "success"
          )
      }
  })
  # 判斷user是不是unique
  # 在上傳圖片時同時做檔名搜尋，如果存在就再取新檔名
  user_img_no<-reactiveValues(num=0)
  observeEvent(input$upload_file,{
      while(file.exists(paste0("tempimg/temp",user_img_no$num,".jpg"))){ 
          user_img_no$num=user_img_no$num+1
      }
      file.copy(from=input$upload_file$datapath # 因為python模組那邊無法讀到虛擬路徑，所以在
                ,to=paste0("tempimg/temp",user_img_no$num,".jpg"))
      # 為了避免上傳了要被判斷的圖，就(1st round)/仍輸出上一輪已產出的圖，要先把資料夾中上一輪的圖刪掉
      # 可以不用再將圖刪掉了
      # if(file.exists("predimg/output_img.jpg")){ 
      #     file.remove("predimg/output_img.jpg")
      # }
  })
}