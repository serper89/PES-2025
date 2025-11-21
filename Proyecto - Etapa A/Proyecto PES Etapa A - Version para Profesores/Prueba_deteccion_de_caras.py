#Face recognition imports
import os
from pathlib import Path
import cv2
import face_recognition
import os.path
#Google Drive API imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
#Audio recording imports
import pyaudio
import wave
#MySQL imports
import mysql.connector
from datetime import datetime
#Telegram imports
import asyncio
from telegram import Bot, KeyboardButton, ReplyKeyboardMarkup
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
# Parametros chat telegram
TOKEN='8424092697:AAG4MtTeh6P6zjc6NEDlYY_T-a0Ae8Z2UmU'
chat_id='1489174748'
message='Timbre'

# Parámetros de la grabación
CHUNK = 1024  # Tamaño del buffer de audio
FORMAT = pyaudio.paInt16  # Formato de audio (16 bits)
CHANNELS = 1  # Número de canales (1 para mono)
RATE = 44100  # Frecuencia de muestreo (muestras por segundo)
RECORD_SECONDS = 5  # Duración de la grabación en segundos
WAVE_OUTPUT_FILENAME = "grabacion.wav" # Nombre del archivo a guardar

#Parametros de la API de Google Drive
# If modifying these scopes, delete the file token.json.
SCOPES = [
  "https://www.googleapis.com/auth/drive.file",
  "https://www.googleapis.com/auth/drive.metadata.readonly",
]
folder_id="1bQxwcDfjigfLRxdSj-5BRxVAyq2CR2Rw"

#Funcion para enviar mensaje por telegram
async def envia_mensaje():
    try:
        bot=Bot(token=TOKEN)
        buttons=[[KeyboardButton("Aceptar")], [KeyboardButton("Rechazar")], [KeyboardButton("Esperar")]]
        await bot.send_photo(chat_id=chat_id, photo=open('portero.jpg', 'rb'))
        await bot.send_audio(chat_id=chat_id, audio=open('puerta.wav', 'rb'))
        await bot.send_message(chat_id=chat_id, text=message, reply_markup=ReplyKeyboardMarkup(buttons))
    except:
        print("Se conecto bot telegram")


#Extraer rostros de las imagenes de la carpeta
script_path = Path(__file__).resolve()
imagesPath = script_path.parent
#print(imagesPath)
#imagesPath="C:\\Users\\sergi\\Documents\\Prueba face recognition\\Prueba deteccion de caras"
print(imagesPath)
if not os.path.exists("faces"):
    os.makedirs("faces")
    print("Nueva carpeta creada")

#detector facial
faceClassif=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
c=0
for imagesName in os.listdir(imagesPath):
    print(imagesName)
    fullpath=os.path.join(imagesPath, imagesName)
    image=cv2.imread(fullpath)
   
    faces=faceClassif.detectMultiScale(image,
                                   scaleFactor=1.1,
                                   minNeighbors=5)
        
    for (x,y,w,h) in faces:
        #cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        face= image[y:y+h,x:x+w]
        face=cv2.resize(face,(150,150))
        #face_fullpath=os.path.join("faces/", str(c),".jpg")
        cv2.imwrite("faces/"+ str(c)+ ".jpg",face)
        c  +=1




#codificar los rostros extraidos

facesEncodings=[]
facesNames=[]

folder_name = "faces" 
imagesFacesPath = imagesPath / folder_name
#print(imagesFacesPath1)
imagesFacesPath=str(imagesFacesPath)
#imagesFacesPath="C:\\Users\\sergi\\Documents\\Prueba face recognition\\Prueba deteccion de caras\\faces"
#print(type(imagesFacesPath))
for fileName in os.listdir(imagesFacesPath):
    image_f=cv2.imread(imagesFacesPath+ "/" + fileName)
    #print(fileName)
    image_f=cv2.cvtColor(image_f,cv2.COLOR_BGR2RGB)

    f_coding=face_recognition.face_encodings(image_f,known_face_locations=[(0,150,150,0)])[0]
    facesEncodings.append(f_coding)
    facesNames.append(fileName.split("_")[0])
    
#print(facesEncodings)
#print(facesNames)

faceClassif=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap=cv2.VideoCapture(0)
fps=cap.get(cv2.CAP_PROP_FPS)
size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

while(True):
    ret,frame = cap.read()
    if(ret==True):
        frame=cv2.flip(frame,1)
        orig=frame.copy()
        faces=faceClassif.detectMultiScale(frame,
                                   scaleFactor=1.1,
                                   minNeighbors=5)
        
        for (x,y,w,h) in faces:
            face=orig[y:y+h,x:x+w]
            face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            actual_face_encoding=face_recognition.face_encodings(face, known_face_locations=[(0, w,h,0)])[0]
            result=face_recognition.compare_faces(facesEncodings, actual_face_encoding)
            if True in result:
                index=result.index(True)
                name =facesNames[index]
                color=(125,220,0)
            else:
                name="Desconocido"
                color=(50,50,255)
            
            cv2.rectangle(frame, (x,y+h),(x+w,y+h+30),color,-1)
            cv2.rectangle(frame, (x,y),(x+w,y+h),color,2)
            cv2.putText(frame,name,(x,y+h+25),2,1,(255,255,255),2,cv2.LINE_AA)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow("Frame",frame)
        if (cv2.waitKey(1)&0xFF==ord("q")):
            cv2.imwrite('portero.jpg',frame)
            break
cap.release()
cv2.destroyAllWindows()

#Api para grabar audio cuando se presiona la tecla 'q'

duracion=3
archivo="puerta.wav"

audio=pyaudio.PyAudio()

stream=audio.open(format=pyaudio.paInt16,channels=1,
					rate=44100,input=True,
					frames_per_buffer=1024)
					
print("Grabando ...")
frames=[]

for i in range(0,int(44100/1024*duracion)):
	data=stream.read(1024)
	frames.append(data)
	
print("Grabacion a terminado ")

stream.stop_stream()
stream.close()
audio.terminate()

waveFile=wave.open(archivo,'wb')
waveFile.setnchannels(1)
waveFile.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
waveFile.setframerate(44100)
waveFile.writeframes(b''.join(frames))
waveFile.close()



#Api google drive para subir imagenes y audio a la nube
#Sino encuentra el archivo token.json, se abrira una pagina para autenticar y creara uno nuevo

try:
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists("token.json"):
      creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
      if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
      else:
        flow = InstalledAppFlow.from_client_secrets_file(
            "credentials.json", SCOPES
        )
        creds = flow.run_local_server(port=0)
      # Save the credentials for the next run
      with open("token.json", "w") as token:
        token.write(creds.to_json())
except:
    print("Se cargaron las credenciales google drive")
try:
  service = build("drive", "v3", credentials=creds)
   # Call the Drive v3 API
  file_metadata = {"name": "portero.jpg", "parents": [folder_id]}
  media = MediaFileUpload(
      "portero.jpg", mimetype="image/jpeg", resumable=True
  )
  # pylint: disable=maybe-no-member
  file = (
      service.files()
      .create(body=file_metadata, media_body=media, fields="id")
      .execute()
  )
  print(f'File ID: "{file.get("id")}".')
  permission = {
      'type': 'anyone',
      'role': 'reader',
  }
  service.permissions().create(fileId=file.get("id"), body=permission).execute()
  file_details = service.files().get(fileId=file.get("id"), fields='webViewLink').execute()
  shareable_urlfoto = file_details.get('webViewLink')
  print(f'Shareable URL: {shareable_urlfoto}')
  file_metadata = {"name": "puerta.wav", "parents": [folder_id]}
  media = MediaFileUpload(
      "puerta.wav", mimetype="audio/wav", resumable=True
  )
  # pylint: disable=maybe-no-member
  file = (
      service.files()
      .create(body=file_metadata, media_body=media, fields="id")
      .execute()
  )
  print(f'File ID: "{file.get("id")}".')
  permission = {
      'type': 'anyone',
      'role': 'reader',
  }
  service.permissions().create(fileId=file.get("id"), body=permission).execute()
  file_details = service.files().get(fileId=file.get("id"), fields='webViewLink').execute()
  shareable_urlaudio = file_details.get('webViewLink')
  print(f'Shareable URL: {shareable_urlaudio}')
except HttpError as error:
  # TODO(developer) - Handle errors from drive API.
  #print(f"An error occurred: {error}")
  print("Se cargaron los archivos al google drive")




#Rostro reconocido esta guardado en la variable name
#Enlace URL de la foto en shareable_urlfoto
#Enlace URL del audio en shareable_urlaudio

#Api para guardar la foto y el audio en la base mysql

#print(shareable_urlaudio)
#print(shareable_urlfoto)

#os.remove("portero.jpg")
#os.remove("puerta.wav")
try:
    #Conexion a la base de datos mysql
    basedatos = mysql.connector.connect(user='root', password='', host='localhost', database='timbrepes', port='3306')
    print(basedatos)
    micursorbasedatos=basedatos.cursor()
    """
    micursorbasedatos.execute("DESCRIBE accesousuario")
    for x in micursorbasedatos:
        print(x)
    """
    if name=="Desconocido":
        Estado="Denegado"
        print("Rostro no reconocido")
        
    else:
        Estado="Aceptado"
        name = name[:-4]
        print("Rostro reconocido:", name) 
    
    #Obtener fecha y hora actual
    fechaactual=datetime.now()
    formato_fechaactual=fechaactual.strftime('%Y-%m-%d %H:%M:%S')
    #Insertar datos en la tabla accesousuario
    micursorbasedatos.execute("INSERT INTO accesousuario (Estado, Usuario, Fecha, EnlaceFoto, EnlaceAudio) VALUES (%s, %s, %s, %s, %s)", (Estado, name, formato_fechaactual, shareable_urlfoto, shareable_urlaudio))
    basedatos.commit()
except:
    print("Se cargaron los datos en la base mysql")    

try:
    if name=="Desconocido":
        print("Rostro no reconocido")
        asyncio.run(envia_mensaje())
    else:
        name = name[:-4]
        print("Rostro reconocido:", name)
        asyncio.run(envia_mensaje())
except:
    print("Se envia mensaje multimedia por telegram") 


