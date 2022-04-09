from keras.models import load_model
import tensorflow as tf
from fastapi import FastAPI,UploadFile,File,Request,status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import numpy as np
import uvicorn


app = FastAPI()

std_bone_age = 127.3207517246848
mean_bone_age = 41.18202139939618

Model = tf.keras.models.load_model("../models/best_model.h5",compile=False)

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(data).convert("RGB").resize((256,256)))
    image = image/255.0
    return image

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
  image = read_file_as_image(await file.read())  
  img_array = tf.expand_dims(image, 0)
  predictions = std_bone_age+mean_bone_age*(Model.predict(img_array,batch_size=32))
  predict = predictions/12.0

  return {"prediction" : float(predict)}

if __name__ == "__main__":
    uvicorn.run(app, host='localhost',port=8000)