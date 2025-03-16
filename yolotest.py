from ultralytics import YOLO

model = YOLO("runs\\detect\\train7\\weights\\best.pt")

# model.predict('E:\\work\\highstakes\\test_2.jpg', save = True)
# model.predict('E:\\work\\highstakes\\test_3.jpg', save = True)
# model.predict('E:\\work\\highstakes\\test_4.jpg', save = True)


model.predict(
    "test_data\\videoplayback_4.mp4", save=True, hide_labels=True, hide_conf=True
)

# model.predict('E:\\work\\highstakes\\maxresdefault.jpg', save = True, hide_labels=True, hide_conf=True)
# model.predict('E:\\work\\highstakes\\training_data_detect_ring\\valid\\images\\image_jpg.rf.b373a82ad9b08e583772f0b505beacf2.jpg', save = True)
