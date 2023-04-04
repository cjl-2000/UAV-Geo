
import cv2
import numpy as np

# 提取超像素块
def extract_superpixels(img, num_segments):
    # 进行超像素分割
    slic = cv2.ximgproc.createSuperpixelSLIC(img, algorithm=cv2.ximgproc.SLICO, region_size=num_segments, ruler=10.0)
    slic.iterate(10)

    # 获取超像素的标签
    labels = slic.getLabels()

    # 提取每个超像素块
    superpixels = []
    for i in range(np.max(labels) + 1):
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        mask[labels == i] = 255
        superpixels.append(cv2.bitwise_and(img, img, mask=mask))

    return superpixels, labels

# 去除位于图像边缘的超像素块
def remove_edge_superpixels(superpixels, labels, threshold):
    edge_labels = set([labels[0, j] for j in range(labels.shape[1])])
    edge_labels.update(set([labels[-1, j] for j in range(labels.shape[1])]))
    edge_labels.update(set([labels[i, 0] for i in range(labels.shape[0])]))
    edge_labels.update(set([labels[i, -1] for i in range(labels.shape[0])]))

    new_superpixels = []
    for i in range(len(superpixels)):
        if i not in edge_labels:
            new_superpixels.append(superpixels[i])

    return new_superpixels

# 去除超像素边缘点作为特征点
def extract_feature_points(img, superpixels, num_points):
    feature_points = []
    for i in range(len(superpixels)):
        for j in range(num_points):
            x = np.random.randint(0, superpixels[i].shape[1])
            y = np.random.randint(0, superpixels[i].shape[0])
            feature_points.append([x + superpixels[i].shape[1] * i, y])
    feature_points = np.array(feature_points)

    # 去除梯度相同的同质区域特征点
    new_feature_points = []
    for i in range(len(feature_points)):
        x, y = feature_points[i]
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_DEFAULT)[y, x]
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_DEFAULT)[y, x]
        gradients = np.sqrt(grad_x**2 + grad_y**2)
        if np.sum(gradients > 500) > 0:
            new_feature_points.append([x, y])
    return np.array(new_feature_points)

# 可视化特征点
def draw_feature_points(img, feature_points):
    for i in range(len(feature_points)):
        x, y = feature_points[i]
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
