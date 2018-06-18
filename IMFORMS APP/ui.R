library(shinydashboard)
library(shiny)
library(shinyWidgets)
library(dplyr)
widget_col_size=3
job_selection<-c("Student","Service","Finace","Medical","Agriculture","Technology","Manufacturing","Retired")

dashboardPage(
    skin = "purple",
    dashboardHeader(
        title="INFORMS HACKTHON",
        tags$li(a(img(src = 'DALab.png', title = "A+ DALab", height = "50px"), 
                  style = "padding-top:0px; padding-bottom:0px;"),
                  class = "dropdown")
        
    ),
    dashboardSidebar(collapsed = F
        ,sidebarMenu(
            menuItem("Introduction", tabName = "Introduction", icon = icon("address-book"))
            ,menuItem("Statistic Analysis", icon = icon("info-circle"), 
                     href = "http://mortality.geohealth.tw")
            ,menuItem("Inspection", icon = icon("search")
                      ,menuSubItem("Melanoma", tabName = "Melanoma",icon = icon("angle-double-right"))
                      ,menuSubItem("jaundice", tabName = "jaundice",icon = icon("angle-double-right"))
                        )
            ,menuItem("Auscultation", tabName = "Auscultation", icon = icon("assistive-listening-systems"))
            ,menuItem("Percussion", tabName = "Percussion", icon = icon("user-md"))
            ,menuItem("Palpation", tabName = "Palpation", icon = icon("ambulance"))
            ,menuItem("Help Us", tabName = "supus", icon = icon("address-book"))
        )
    ),
    dashboardBody(
      includeCSS("www/custom.css"),
        #### UI settings ####
      #控制progress bar的CSS
        tags$head(
          tags$style(
              HTML(".shiny-notification {
                   height: 100px;
                   width: 800px;
                   position:fixed;
                   top: calc(50% - 50px);
                   left: calc(50% - 400px);
                   font-size: 250%;
                   text-align: center;
                   }
                   "
              )
          )
        ),
        tags$head(
          tags$link(rel = "stylesheet", type = "text/css", href = "custom.css")
        )
        ,tags$style(HTML('
        hr{display:block; height:1px; border:0; border-top:2px solid #ABABAB; margin:1em 0; padding: 0;}'
        )),
        tags$head(tags$style(HTML(
            '.myClass {font-size:20px; line-height:50px; text-align:left; font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;
          padding:0 15px; overflow:hidden; color:white;}'
        ))),
        tags$script(HTML('
          $(document).ready(function() {
          $("header").find("nav").append(\'<span class="myClass"><span style="font-family:Georgia;font-weight: bold">AI In Healthcare</span>\');})'
        )),
        ####
        tabItems(
            tabItem("Introduction",fluidRow(
              column(12,htmlOutput("toptext"))
              ,column(5,h6(),imageOutput("textimage",height = "600px",width = "300px"))
              ,column(7,h4(),htmlOutput("text1"),h3())
              ,column(7,h1(),htmlOutput("text2"))
              ,column(7,htmlOutput("text3"))
              ,column(7,htmlOutput("text4"))
              ,column(12,h4(),htmlOutput("bottomtext"))
            ))
            # Melanoma ----
            ,tabItem(tabName = "Melanoma",
                     fluidRow(
                       column(width=4,
                              shinydashboard::box(
                                width = NULL,
                                fileInput(inputId = 'upload_file', 
                                          label = h4('Upload an Image'),
                                          #multiple = TRUE,
                                          accept=c('image/png', 'image/jpeg'))
                                ,column(12,align="center",actionBttn("python", label='Go!', color="primary", size="lg"))
                                
                              ),
                              shinydashboard::box(
                                  width = NULL,
                                  # collapsible = TRUE,
                                  uiOutput("upload_img")
                                  ,htmlOutput("suggestion")
                              )
                       ),
                       column(width=8, 
                              shinydashboard::box(width='400', height = "600"
                                                  ,column(12,align="center",htmlOutput("Bigsuggestion"))
                                                  ,column(12,align="center",h1(),uiOutput("pred_img"))
                              )
                       )
                     )     
            )
            ,tabItem("jaundice",HTML("<font size=\"9\"><b>To be continued</font></b>"))
            ,tabItem("Auscultation",HTML("<font size=\"9\"><b>To be continued</font></b>"))
            ,tabItem("Percussion",HTML("<font size=\"9\"><b>To be continued</font></b>"))
            ,tabItem("Palpation",HTML("<font size=\"9\"><b>To be continued</font></b>"))
            ,tabItem("supus"
                     ,awesomeRadio(
                       inputId = "Gender", label =HTML("<font size=\"5\">Gender</font>"), 
                       choices = c("Male","Female"),checkbox = TRUE
                     )%>%column(widget_col_size,align="center",.)
                     ,knobInput(
                       inputId = "Height",label = HTML("<font size=\"5\">Height</font>"),
                       value = 175,min = 100,max = 250,step=0.1,
                       displayPrevious = TRUE,width=80,
                       lineCap = "round",fgColor = "#428BCA",
                       inputColor = "#428BCA"
                     )%>%column(widget_col_size,align="center",.)
                     ,knobInput(
                       inputId = "Weight",label = HTML("<font size=\"5\">Weight</font>"),
                       value = 30,min = 0,max = 200,step=0.1,
                       displayPrevious = TRUE,width=80,
                       lineCap = "round",fgColor = "#ff0000",
                       inputColor = "#ff0000"
                     )%>%column(widget_col_size,align="center",.)
                     ,knobInput(
                       inputId = "Age",label = HTML("<font size=\"5\">Age</font>"),
                       value = 18,min = 0,max = 150,step=1,
                       displayPrevious = TRUE,width=80,
                       lineCap = "round",fgColor = "#ff9900",
                       inputColor = "#ff9900"
                     )%>%column(widget_col_size,align="center",.)
                     ,awesomeRadio(
                       inputId = "Smoking", label =HTML("<font size=\"5\">Smoking</font>"), 
                       choices = c("Yes","No"),checkbox = TRUE,status = "danger"
                     )%>%column(widget_col_size,align="center",.)
                     ,awesomeRadio(
                       inputId = "Drinking", label =HTML("<font size=\"5\">Drinking</font>"), 
                       choices = c("Yes","No"),checkbox = TRUE,status = "success"
                     )%>%column(widget_col_size,align="center",.)
                     ,pickerInput(
                       inputId = "Job",
                       label = HTML("<font size=\"5\">Job</font>"), 
                       choices = job_selection,
                       options = list(
                         `live-search` = TRUE)
                     )%>%column(widget_col_size,align="center",.)
                     ,textInput(inputId = "Email",
                       label = HTML("<font size=\"5\">Email</font>")
                     )%>%column(widget_col_size,align="center",.)
                     ,actionBttn(
                       inputId = "launch",
                       label = "Send Out",
                       style = "pill", 
                       color = "danger"
                     )%>%column(12,h1(),align="center",.)
             )
        )
    )
)