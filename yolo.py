from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model.train(data="data.yaml", epochs=30)

results = model.predict(["choc.jpg", "dop.jpg"])

for result in results:
    # Now, 'result' contains the prediction results for each image
    print("Number of boxes:", len(result.boxes))
    for box in result.boxes:
        class_id = result.names[box.cls[0].item()]
        coords = box.xyxy[0].tolist()
        coords = [round(x) for x in coords]
        conf = round(box.conf[0].item(), 2)
        print("Object type:", class_id)
        print("Coordinates:", coords)
        print("Probability:", conf)
        print("---")





