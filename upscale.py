import cv2
from cv2 import dnn_superres
sr = dnn_superres.DnnSuperResImpl_create()
image = cv2.imread('000000001298.png')
print(image.shape)
path = "ESPCN_x3.pb"
sr.readModel(path)
sr.setModel("espcn", 3)
result = sr.upsample(image)
# Save the image
cv2.imwrite("upscaled.png", result)