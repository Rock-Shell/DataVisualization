import cv2
import pandas as pd
import numpy as np

# colors for different parts and damages
colors = {}

for img_no in range(1,6):
    data = pd.read_json(f"data/{img_no}.json")
    img = cv2.imread(f"images/{img_no}.jpg")
    blackie = np.zeros(img.shape, np.uint8)
    w,h = data.iloc[1].original_width, data.iloc[1].original_height
    for i in range(1,data.shape[0]):
        pts = []
        part = data.value[i]['polygonlabels'][0]
        for x,y in data.value[i]['points']:
            x = x*w//100
            y = y*h//100
            pts.append([x,y])

        pts = np.array(pts,np.int32).reshape((-1,1,2))
        if part in colors.keys():
            color = colors[part]
        else:
            color = tuple(np.random.choice(range(256), size=3))
            color = (int(color[0]), int(color[1]), int(color[2]))
            colors[part] = color
        cv2.drawContours(blackie, [pts],-1, color,-1)
    out = cv2.addWeighted(img, 0.5, blackie, 0.5,0.0)
    cv2.imshow("img",out)
    cv2.imshow("blackie",blackie)
    cv2.waitKey(0)

cv2.destroyAllWindows()
