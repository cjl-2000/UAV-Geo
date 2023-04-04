import cv2
import numpy as np


def slic(img, k):
    # Slic超像素分割
    slic = cv2.ximgproc.createSuperpixelSLIC(img, algorithm='SLIC', region_size=20, ruler=10.0)
    slic.iterate(k)
    slic_enforce_connectivity = True
    if slic_enforce_connectivity:
        slic.enforceLabelConnectivity()

    # 生成超像素块边缘图
    mask = slic.getLabelContourMask()

    return slic, mask


def get_edge_points(img, mask):
    # 获取超像素块的边缘特征点
    edge_points = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 255:
                # 剔除与同质区域相似的边缘点
                if abs(np.gradient(img[i, j])[0]) > 10 or abs(np.gradient(img[i, j])[1]) > 10:
                    edge_points.append((i, j))
    return edge_points


def knn_match(edge_points1, edge_points2, k=2):
    # k最近邻匹配
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    descriptors1 = np.zeros((len(edge_points1), 32), np.uint8)
    descriptors2 = np.zeros((len(edge_points2), 32), np.uint8)

    for i in range(len(edge_points1)):
        x, y = edge_points1[i]
        patch = cv2.resize(img1[x - 8:x + 8, y - 8:y + 8], (4, 4), interpolation=cv2.INTER_AREA)
        descriptors1[i] = cv2.KeyPoint(8, 8, _size=16).compute(patch).descriptors

    for i in range(len(edge_points2)):
        x, y = edge_points2[i]
        patch = cv2.resize(img2[x - 8:x + 8, y - 8:y + 8], (4, 4), interpolation=cv2.INTER_AREA)
        descriptors2[i] = cv2.KeyPoint(8, 8, _size=16).compute(patch).descriptors

    matches = matcher.knnMatch(descriptors1, descriptors2, k)

    good_matches = []
    for i in range(len(matches)):
        if len(matches[i]) == 2 and matches[i][0].distance < 0.8 * matches[i][1].distance:
            good_matches.append((edge_points1[matches[i][0].queryIdx], edge_points2[matches[i][0].trainIdx]))

    return good_matches


# 加载图像
img1 = cv2.imread('1.jpg')
img2 = cv2.imread('2.jpg')

# Slic超像素分割
slic1, mask1 = slic(img1, k=200)
slic2, mask2 = slic(img2, k=200)

# 获取边缘特征点
edge_points1 = get_edge_points(img1, mask1)
edge_points2 = get_edge_points(img2, mask2)

# k最近邻匹配
matches = knn_match(edge_points1, edge_points2, k=2)

# 输出匹配结果
for match in matches:
    print(match)