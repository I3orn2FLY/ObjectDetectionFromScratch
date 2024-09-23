import cv2
import numpy
import json
import os

if __name__ == "__main__":

    with open("data/annotations/instances_val2017.json", 'r') as f:
        anno_json = json.load(f)

    annos = anno_json["annotations"]

    img_id2bboxes = {}
    for item in annos:
        img_id = item["image_id"]
        bbox = item["bbox"]
        label = item["category_id"]
        bboxes = img_id2bboxes.get(img_id, [])
        bboxes.append(bbox + [label])
        img_id2bboxes[img_id] = bboxes

    path_prefix = "data/val2017"
    for img_info in anno_json["images"]:
        img_path = os.path.join(path_prefix, img_info["file_name"])

        img = cv2.imread(img_path)

        bboxes = img_id2bboxes[img_info["id"]]

        for box in bboxes:
            x,y,w,h = [int(i) for i in box[:-1]]
            label = box[-1]

            cv2.putText(img, str(label), (x, y), cv2.FONT_HERSHEY_PLAIN, 2, color=(255,0,255))
            cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 255), 2)


        # cv2.rectangle(img, tuple(int(bboxes[0])))
        cv2.imshow("Window", img)
        cv2.waitKey(0)


    # img = cv2.imread()