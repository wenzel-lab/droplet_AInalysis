if __name__ == "__main__":
    from ultralytics import YOLO
    from os.path import join
    import torch


    torch.cuda.empty_cache()
    torch.device("cuda:0")
    model = YOLO(join("runs","detect","test12","weights","best.pt"))

    results = model.train(data="train_1.yaml", 
                          single_cls = True,
                          epochs=150, workers=1, 
                          imgsz=640, batch=9, 
                          name="test13", 
                          device="0", 
                          close_mosaic=0, 
                          max_det=400)