import cv2
import pytesseract
image = cv2.imread('number_plate.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
text = pytesseract.image_to_string(image)
print (text)
