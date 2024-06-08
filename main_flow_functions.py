from time import sleep

def waiting_screen(evento):
    i = 0
    while not evento.is_set():
        print(f"Seting up the model{'.'*i}", end='\r', flush=True)
        sleep(0.8)
        i+=1
    print("Model ready to predict!")

def set_up(evento, queue):
    from os.path import join
    from ultralytics import YOLO as Yolo
    from PARAMETERS import WEIGHT, IMGSZ, MAX_DETECT

    model = Yolo(WEIGHT)
    model.predict(join("testing_imgs","none.jpg"), imgsz=IMGSZ, max_det=MAX_DETECT)
    queue.put(model)

    evento.set()
    return model

def predict_v1(model):
    pass

def predict_v2():
    pass