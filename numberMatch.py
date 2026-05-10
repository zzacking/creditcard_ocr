import cv2
import numpy as np
import os


def show(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def stable(data):
    find = 0
    offset = 3
    for i in range(0, len(data)):
        if abs(data[i + 1] - data[i]) <= offset:
            find += 1
        else:
            find = 0
        if find == 3:
            return i - 2
    return -1


# template loading
template = cv2.imread("../../img_collection/number_ref.png")
gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
binary_copy = binary.copy()
contours, hierarchy = cv2.findContours(binary_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(template, contours, -1, (0, 0, 255), 2)
bounding_boxes = [cv2.boundingRect(c) for c in contours]
contours, bounding_boxes = zip(*sorted(zip(contours, bounding_boxes), key=lambda x: x[1][0]))

dic = {}
for i, c in enumerate(contours):
    x, y, w, h = cv2.boundingRect(c)
    roi = binary[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))
    dic[i] = roi

# 获取所有信用卡图片
img_dir = "../../img_collection/"
credit_cards = sorted([f for f in os.listdir(img_dir) if f.startswith("creditcard") and f.endswith(".png")])
print(credit_cards)
if not credit_cards:
    print("no imgs！")
    exit()

current_index = 0
print(f"find {len(credit_cards)} imgs")
print("D/d next | A/a previous | q/ESC quit")

while True:
    # 读取当前图片
    img_path = os.path.join(img_dir, credit_cards[current_index])
    print(f"dealing: {credit_cards[current_index]}")

    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    width_set = 500
    r = width_set / w
    img = cv2.resize(img, (width_set, int(h * r)))

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 3))
    tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
    #Sobel Algo + otsu to find four number blocks
    grad_x = cv2.Sobel(tophat, cv2.CV_32F, dx=1, dy=0, ksize=3)
    grad_x = np.absolute(grad_x)
    min_val = np.min(grad_x)
    max_val = np.max(grad_x)
    grad_x = ((grad_x - min_val) / (float(max_val) - float(min_val))) * 255
    grad_x = grad_x.astype(np.uint8)

    big_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, big_kernel, iterations=3)

    _, dst = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, big_kernel, iterations=3)

    contours, _ = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    locs = []
    output = []
    width_list = []
    height_list = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        width_list.append(w)
        height_list.append(h)
    # the four number blocks have a relatively stable width, so if we find
    # 4 consecutive data, we could identify it as the roi
    start_index = stable(width_list)
    height_list = height_list[start_index:start_index + 4]

    #remove the maximum and the minimum to get average
    height_list.remove(max(height_list))
    height_list.remove(min(height_list))

    avg = int(np.mean(height_list))

    for c in contours[start_index:start_index + 4]:
        x, y, w, h = cv2.boundingRect(c)
        if abs(h - avg) > 3: # rectify
            h = avg
        locs.append((x, y, w, h))

    locs = sorted(locs, key=lambda x: x[0])

    for i, (gx, gy, gw, gh) in enumerate(locs):
        group = img_gray[gy - 5:gy + gh + 5, gx - 5:gx + gw + 5]
        _, group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        digit_contours, _ = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(c) for c in digit_contours]
        digit_contours, bounding_boxes = zip(*sorted(zip(digit_contours, bounding_boxes), key=lambda b: b[1][0]))

        out_put = []
        for c in digit_contours:
            x, y, w, h = cv2.boundingRect(c)
            roi = group[y:y + h, x:x + w]
            roi = cv2.resize(roi, (57, 88))

            scores = []
            for _, roi_digit in dic.items():
                result = cv2.matchTemplate(roi, roi_digit, cv2.TM_CCOEFF_NORMED)
                score = cv2.minMaxLoc(result)[1]
                scores.append(score)

            out_put.append(str(np.argmax(scores)))

        cv2.rectangle(img, (gx - 5, gy - 5), (gx + gw + 5, gy + gh + 5), (0, 0, 255), 2)
        cv2.putText(img, ''.join(out_put), (gx, gy - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        output.extend(out_put)

    # show the result
    cv2.putText(img,f"{current_index+1}",(10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    cv2.imshow('Credit Card Recognition', img)


    # keyboard control
    key = cv2.waitKey(0) & 0xFF

    if key == ord('q') or key == 27:  # q or ESC to quit
        break
    elif key == ord('d') or key == ord('D'):  # D or d next
        current_index = (current_index + 1) % len(credit_cards)
    elif key == ord('a') or key == ord('A'):  # A or a previous
        current_index = (current_index - 1) % len(credit_cards)

cv2.destroyAllWindows()
print("over")