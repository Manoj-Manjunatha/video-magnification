import cv2

img = cv2.imread("bird-thumbnail.jpg")
cv2.imshow('img', img)

img_level_1 = cv2.pyrDown(img)
img_level_2 = cv2.pyrDown(img_level_1)

# cv2.imshow('img_level_1', img_level_1)
# cv2.imshow('img_level_2', img_level_2)

# cv2.waitKey(0)



lower = img.copy()
 
# Create a Gaussian Pyramid
gaussian_pyr = [lower]
for i in range(3):
    lower = cv2.pyrDown(lower)
    gaussian_pyr.append(lower)
 
# Last level of Gaussian remains same in Laplacian
laplacian_top = gaussian_pyr[-1]
 
# Create a Laplacian Pyramid
laplacian_pyr = [laplacian_top]
for i in range(3, 0, -1):
    size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
    gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)
    laplacian = cv2.subtract(gaussian_pyr[i - 1], gaussian_expanded)
    laplacian_pyr.append(laplacian)
    cv2.imshow('lap-{}'.format(i - 1), laplacian)
    cv2.waitKey(0)

cv2.destroyAllWindows()
