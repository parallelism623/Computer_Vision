import cv2
from cv2 import dnn_superres
def up_scale_x3(path_image, save_path):
    sr = dnn_superres.DnnSuperResImpl_create()
    image = cv2.imread(path_image)
    print(image.shape)
    path = "ESPCN_x3.pb"
    sr.readModel(path)
    sr.setModel("espcn", 3)
    result = sr.upsample(image)
    # Save the image
    cv2.imwrite(save_path, result)
