# import the necessary packages
import os

import click
import cv2
import ocrmypdf
import pytesseract
from PIL import Image
from spellchecker import SpellChecker


class performOCR:
    def performOCR(self, imagePath, outputPath):
        self.imagepath = imagePath
        self.outputPath = outputPath
        self.img = None

    # https://becominghuman.ai/how-to-automatically-deskew-straighten-a-text-image-using-opencv-a0c30aed83df
    def getSkewAngle(self, img) -> float:
        # Prep image, copy, convert to gray scale, blur, and threshold
        newImage = img.copy()
        gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Apply dilate to merge text into meaningful lines/paragraphs.
        # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
        # But use smaller kernel on Y axis to separate between different blocks of text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilate = cv2.dilate(thresh, kernel, iterations=2)

        # Find all contours
        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in contours:
            rect = cv2.boundingRect(c)
            x, y, w, h = rect
            cv2.rectangle(newImage, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Find largest contour and surround in min area box
        largestContour = contours[0]
        # print(len(contours))
        minAreaRect = cv2.minAreaRect(largestContour)
        cv2.imwrite("temp/boxes.jpg", newImage)
        # Determine the angle. Convert it to the value that was originally used to obtain skewed image
        angle = minAreaRect[-1]
        if angle < -45:
            angle = 90 + angle
        return -1.0 * angle

    # Rotate the image around its center
    def rotateImage(self, img, angle: float):
        newImage = img.copy()
        (h, w) = newImage.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return newImage

    # Deskew image
    def deskew(self, img):
        angle = self.getSkewAngle(img)
        return self.rotateImage(img, -1.0 * angle)

    def thin_font(self, img):
        import numpy as np
        image = cv2.bitwise_not(img)
        kernel = np.ones((2, 2), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.bitwise_not(image)
        return (image)

    def loadImage(self):
        img = cv2.imread(self.imagepath)
        return img

    def isPdf(self):
        if ".pdf" in self.imagepath:
            return True
        return False

    def ocrOnPDF(self):
        if ".pdf" != self.outputPath[-4:]:
            self.outputPath = self.outputPath+".pdf"
        ocrmypdf.ocr(self.imagepath, self.outputPath, deskew=True)

    def saveTextFile(self, text):
        textfile = open(self.outputPath, 'w')
        textfile.write(text)
        textfile.close()

    def ocrOnImages(self):
        self.img = self.loadImage()
        fixed = self.deskew(self.img)

        filename = "thin_image.jpg"

        eroded_image = self.thin_font(fixed)
        cv2.imwrite(filename, eroded_image)

        text = pytesseract.image_to_string(Image.open(filename))
        os.remove(filename)

        spell = SpellChecker()
        words = str.split(text)
        test = [spell.correction(word) for word in words]

        self.saveTextFile(" ".join(test))

    def startOCR(self):
        if ".pdf" == self.imagepath[-4:]:
            self.ocrOnPDF()
            return True
        else:
            return self.ocrOnImages()



imagePath = "samples/82251504.png"
outputPath = "output.txt"

@click.command()
@click.option('--input', default="test.png", help='Enter the path of input file')
@click.option('--output', default='output.txt',
              help='Output Path')
@click.option('--verbose', default=1,
              help='Verbose')

def runOcr(input, output, verbose):
    """Simple program that greets NAME for a total of COUNT times."""
    pOcr = performOCR()
    pOcr.performOCR(input, output)
    pOcr.startOCR()

if __name__ == '__main__':
    runOcr()