import streamlit as st
from collections import Counter
from PIL import Image
from ultralytics import YOLO
def process_image(image, conf, model):
    """
    this function will process the image and return the detected objects
    inputs:
    image(PIL.Image.Image) -------> PIL image object
    conf(float) ------------------> confidence level ranges from 0 to 1
    outputs:
    result(List[ultralytics.engine.results.Results]): list of results used for the purpose of visualization
    objects(List[str]): list of detected objects
    """
    Objects = []
    result  = model.predict(image,conf = conf)[0]
    boxes   = result.boxes
    for box in boxes:
      Object = result.names[box.cls[0].item()]
      Objects.append(Object)
    return result, Objects

def main(model):
    st.title("Object Detection")
    uploaded_image = st.file_uploader("Upload your Image")
    conf = st.slider("Select the level of confidence", 0.0, 1.0,value = 0.8, step = 0.01)
    # click the button to show the ouput
    # as unordered list of detected objects then plot the image
    if(st.button("Show results")):
      if uploaded_image:
        image = Image.open(uploaded_image)
        result, objects = process_image(image, conf, model)
        objects = Counter(objects)
        st.header("Objects detected:")
        for object_type, count in objects.items():
          st.markdown(f"- {count} {object_type} objects")
        st.image(Image.fromarray(result.plot()[:,:,::-1]))
      else:
        st.error("please upload an image to process")
if __name__ == '__main__':
    # load/download the yolo model
    model = YOLO("yolov8m.pt")
    main(model)
