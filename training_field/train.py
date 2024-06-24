if __name__ == "__main__":
    from ultralytics import YOLO
    from os.path import join
    import torch


    torch.cuda.empty_cache()
    torch.device("cuda:0")
    model = YOLO(join("runs","detect","test11","weights","best.pt"))

    results = model.train(data="train_1.yaml", 
                          epochs=120, workers=1, 
                          imgsz=640, batch=9, 
                          name="test12", 
                          device="0", 
                          close_mosaic=0, 
                          max_det=400)