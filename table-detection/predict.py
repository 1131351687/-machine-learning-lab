from ultralytics import YOLO
import cv2

if __name__ == "__main__":
    model = YOLO("runs/detect/train-2/weights/best.pt")
    result=model.predict(['test1.jpg'],conf=0.1,save=True)
    result[0].show()

    # results = model.val(verbose=False)
    # print(f"\n📊 总体性能:")
    # print(f"   mAP@0.5:     {results.box.map50:.3f}")
    # print(f"   mAP@0.5:0.95: {results.box.map:.3f}")
    # print(f"   精确率 (P):   {results.box.p[0]:.3f}")
    # print(f"   召回率 (R):   {results.box.r[0]:.3f}")
    #
    # print(f"\n📈 各类别 mAP@0.5:")
    # for i, ap in enumerate(results.box.ap50):
    #     if i < len(model.names):
    #         name = model.names[i]
    #     else:
    #         name = f"class_{i}"
    #     print(f"  {name}: {ap:.3f}")