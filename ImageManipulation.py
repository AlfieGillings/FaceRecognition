
# Imported libraries
from PIL import Image, ImageDraw
import math as maths
import time
import os

# Function to resize an image by a given factor
def resizeImage(image, factor):

    # Define variables
    result = image.copy()
    width, height = image.size

    # Resize image
    result = result.resize((int(width * factor), int(height * factor)))

    return result

# Function to convert an image to greyscale
def greyscale(image):

    output = image.copy()
    pixels = output.load()

    # Calculate and set luminance for each pixel
    for i in range(output.size[0]):
        for n in range(output.size[1]):

            r = pixels[i, n][0]
            g = pixels[i, n][1]
            b = pixels[i, n][2]

            luminance = maths.floor((r * 0.2126) + (g * 0.7152) + (b * 0.0722))
            pixels[i, n] = (luminance, luminance, luminance)

    return output

# Function to blur an image
def blur(image, kernelRadius = 1):

    output = image.copy()
    pixels = output.load()

    # Initialise kernel
    kernelSize = (kernelRadius * 2) + 1
    kernel = [[0] * kernelSize for i in range(kernelSize)]

    # Calculate kernel sides
    for i in range(kernelSize):
        pascalValue = maths.floor(maths.factorial(kernelSize - 1) / (maths.factorial(i) * maths.factorial((kernelSize - 1) - i)))
        kernel[i][0] = pascalValue
        kernel[0][i] = pascalValue

    # Fill kernel
    for i in range(kernelSize - 1):
        for n in range(kernelSize - 1):
            kernel[i + 1][n + 1] = kernel[i + 1][0] * kernel[0][n + 1]

    blurred = [[0] * output.size[1] for i in range(output.size[0])]
    # For each pixel
    for i in range(output.size[1]):
        for n in range(output.size[0]):
            X = i - kernelRadius - 1
            Y = n - kernelRadius - 1
            total = 0
            denominator = 0

            # Apply the kernel
            for x in range(kernelSize):
                X = X + 1
                if X > 0 and X < output.size[1] - 1:
                    for y in range(kernelSize):
                        Y = Y + 1
                        if Y > 0 and Y < output.size[0] - 1:
                            total = total + (pixels[Y, X][0] * kernel[y][x])
                            denominator = denominator + kernel[y][x]

            blurred[n][i] = maths.floor(total / denominator)

    for i in range(len(blurred)):
        for n in range(len(blurred[i])):
            value = blurred[i][n]
            pixels[i, n] = (value, value, value)

    return output

# Function to detect edges in an image
def edgeDetection(image, sensitivity):

    output = image.copy()
    pixels = output.load()
    sensitivity = 255 - sensitivity
    edges = [[0] * output.size[1] for i in range(output.size[0])]

    # Check each pixel for edges
    for i in range(output.size[0]):
        for n in range(output.size[1]):

            if i < output.size[0] - 1:
                if abs(pixels[i, n][0] - pixels[i + 1, n][0]) > sensitivity:
                    edges[i][n] = 1
            if i > 0:
                if abs(pixels[i, n][0] - pixels[i - 1, n][0]) > sensitivity:
                    edges[i][n] = 1
            if n < output.size[1] - 1:
                if abs(pixels[i, n][0] - pixels[i, n + 1][0]) > sensitivity:
                    edges[i][n] = 1
            if n > 0:
                if abs(pixels[i, n][0] - pixels[i, n - 1][0]) > sensitivity:
                    edges[i][n] = 1

    # Highlight edges
    for i in range(len(edges)):
        for n in range(len(edges[i])):
            if edges[i][n] == 1:
                pixels[i, n] = (255, 255, 255)
            else:
                pixels[i, n] = (0, 0, 0)

    return output

# Function to generate an integral image
def generateIntegral(image):

    # Define variables
    width, height = image.size
    pixels = image.load()
    integral = [[0] * width for i in range(height)]

    # Iterate through each integral element
    for i in range(height):
        for n in range(width):
            # Calculate value
            a = integral[i - 1][n]
            b = integral[i][n - 1]
            c = integral[i - 1][n - 1]
            if (i == 0):
                a = 0
                c = 0
            if (n == 0):
                b = 0
                c = 0
            integral[i][n] = pixels[n, i][0] + a + b - c

    return integral

# Function to apply a feature to an image using an integral image
def applyFeature(feature, integral, frameStart, frameSize):

    # Define variables
    result = 0
    start = [frameStart[0] + int(frameSize * (feature["start"][0] / 24)), frameStart[1] + int(frameSize * (feature["start"][1] / 24))]
    size = [int(frameSize * (feature["size"][0] / 24)), int(frameSize * (feature["size"][1] / 24))]

    # Apply feature to image
    if (feature["type"] == "horizontal_edge"):

        v1 = integral[start[1] - 1][start[0] - 1]
        v2 = integral[start[1] - 1][int(start[0] + (size[0] * 0.5)) - 1]
        v3 = integral[start[1] - 1][start[0] + size[0] - 1]
        v4 = integral[start[1] + size[1]][start[0] - 1]
        v5 = integral[start[1] + size[1]][int(start[0] + (size[0] * 0.5)) - 1]
        v6 = integral[start[1] + size[1]][start[0] + size[0] - 1]

        if (start[0] == 0):
            v1 = 0
            v4 = 0
        if (start[1] == 0):
            v1 = 0
            v2 = 0
            v3 = 0

        result = (v5 - v4 - v2 + v1) - (v6 - v5 - v3 + v2)

    elif (feature["type"] == "vertical_edge"):

        v1 = integral[start[1] - 1][start[0] - 1]
        v2 = integral[start[1] - 1][start[0] + size[0]]
        v3 = integral[int(start[1] + (size[1] * 0.5)) - 1][start[0] - 1]
        v4 = integral[int(start[1] + (size[1] * 0.5)) - 1][start[0] + size[0]]
        v5 = integral[start[1] + size[1] - 1][start[0] - 1]
        v6 = integral[start[1] + size[1] - 1][start[0] + size[0]]

        if (start[0] == 0):
            v1 = 0
            v3 = 0
            v5 = 0
        if (start[1] == 0):
            v1 = 0
            v2 = 0

        result = (v4 - v3 - v2 + v1) - (v6 - v5 - v4 + v3)

    elif (feature["type"] == "horizontal_line"):
        
        v1 = integral[start[1] - 1][start[0] - 1]
        v2 = integral[start[1] - 1][int(start[0] + (size[0] * (1 / 3))) - 1]
        v3 = integral[start[1] - 1][int(start[0] + (size[0] * (2 / 3))) - 1]
        v4 = integral[start[1] - 1][start[0] + size[0] - 1]
        v5 = integral[start[1] + size[1]][start[0] - 1]
        v6 = integral[start[1] + size[1]][int(start[0] + (size[0] * (1 / 3))) - 1]
        v7 = integral[start[1] + size[1]][int(start[0] + (size[0] * (2 / 3))) - 1]
        v8 = integral[start[1] + size[1]][start[0] + size[0] - 1]

        if (start[0] == 0):
            v1 = 0
            v5 = 0
        if (start[1] == 0):
            v1 = 0
            v2 = 0
            v3 = 0
            v4 = 0

        result = ((v6 - v5 - v2 + v1) * 0.5) - (v7 - v6 - v3 + v2) + ((v8 - v7 - v4 + v3) * 0.5)

    elif (feature["type"] == "vertical_line"):
        
        v1 = integral[start[1] - 1][start[0] - 1]
        v2 = integral[start[1] - 1][start[0] + size[0]]
        v3 = integral[int(start[1] + (size[1] * (1 / 3))) - 1][start[0] - 1]
        v4 = integral[int(start[1] + (size[1] * (1 / 3))) - 1][start[0] + size[0]]
        v5 = integral[int(start[1] + (size[1] * (2 / 3))) - 1][start[0] - 1]
        v6 = integral[int(start[1] + (size[1] * (2 / 3))) - 1][start[0] + size[0]]
        v7 = integral[start[1] + size[1] - 1][start[0] - 1]
        v8 = integral[start[1] + size[1] - 1][start[0] + size[0]]

        if (start[0] == 0):
            v1 = 0
            v3 = 0
            v5 = 0
            v7 = 0
        if (start[1] == 0):
            v1 = 0
            v2 = 0

        result = ((v4 - v3 - v2 + v1) * 0.5) - (v6 - v5 - v4 + v3) + ((v8 - v7 - v6 + v5) * 0.5)

    elif (feature["type"] == "square"):

        v1 = integral[start[1] - 1][start[0] - 1]
        v2 = integral[start[1] - 1][int(start[0] + (size[0] * 0.5)) - 1]
        v3 = integral[start[1] - 1][start[0] + size[0] - 1]
        v4 = integral[int(start[1] + (size[1] * 0.5)) - 1][start[0] - 1]
        v5 = integral[int(start[1] + (size[1] * 0.5)) - 1][int(start[0] + (size[0] * 0.5)) - 1]
        v6 = integral[int(start[1] + (size[1] * 0.5)) - 1][start[0] + size[0] - 1]
        v7 = integral[start[1] + size[1] - 1][start[0] - 1]
        v8 = integral[start[1] + size[1] - 1][int(start[0] + (size[0] * 0.5)) - 1]
        v9 = integral[start[1] + size[1] - 1][start[0] + size [0] - 1]

        if (start[0] == 0):
            v1 = 0
            v4 = 0
            v7 = 0
        if (start[1] == 0):
            v1 = 0
            v2 = 0
            v3 = 0

        result = (v5 - v4 - v2 + v1) - (v6 - v5 - v3 + v2) - (v8 - v7 - v5 + v4) + (v9 - v8 - v6 + v5)

    # Inverted check
    if (feature["inverted"]):
        result = result * (-1)

    return result

# Function to detect faces in an image
def detectFaces(featuresDirectory, image, sizeStep, moveStep, sizeLimit, featureThreshold):

    # Define variables
    start = time.time()
    highestScore = 0
    highestFace = [[0, 0], 0]
    facesNumber = 0
    faces = []
    imageCopy = image.copy()
    draw = ImageDraw.Draw(image)
    greyscaleImage = greyscale(image)
    integral = generateIntegral(greyscaleImage)

    # Get features from file
    features = []
    f = open(featuresDirectory, "r")
    for i in range(featureThreshold):
        feature = f.readline()
        feature = feature.split("|")
        del feature[-1]

        # Convert raw data to dictionary
        feature = {
            "type": feature[0],
            "inverted": eval(feature[1]),
            "start": list(map(int, feature[2].replace("[", "").replace("]", "").replace(" ", "").split(","))),
            "size": list(map(int, feature[3].replace("[", "").replace("]", "").replace(" ", "").split(",")))
        }

        features.append(feature)
    f.close()

    # Set initial frame size
    width, height = greyscaleImage.size
    size = width
    if (height < width):
        size = height

    # Iterate through frame sizes
    while (size >= sizeLimit):
        
        # Iterate through frame positions
        x = 0
        while (x < width - size):
            y = 0
            while (y < height - size):

                # Iterate through features
                score = 0
                for i in range(len(features)):
                    score = applyFeature(features[i], integral, [x, y], size)
                    if (score <= 0):
                        break

                if (score > 0):
                    draw.rectangle([(x, y), (x + size, y + size)], outline = "red", width = 3)
                    faces.append(imageCopy.crop((x, y, x + size, y + size)))
                    facesNumber = facesNumber + 1
                else:
                    if (i + 1 > highestScore):
                        highestScore = i + 1
                        highestFace = [[x, y], size]

                y = y + moveStep
            x = x + moveStep

        size = size - sizeStep

    if (facesNumber == 0):
        draw.rectangle([(highestFace[0][0], highestFace[0][1]), (highestFace[0][0] + highestFace[1], highestFace[0][1] + highestFace[1])], outline = "red", width = 3)
        faces.append(imageCopy.crop((highestFace[0][0], highestFace[0][1], highestFace[0][0] + highestFace[1], highestFace[0][1] + highestFace[1])))
    print(" > Highest: ", str(highestScore))
    print(" > Found ", facesNumber, " faces in " + str(int(time.time() - start)) + " seconds.")

    return [image, faces]

# Function to generate a vector for an image
def generateVector(image):

    # Define variables
    width, height = image.size
    pixels = image.load()
    vector = [0] * (width * height)

    # Generate vector
    for i in range(width):
        for n in range(height):
            vector[(i * height) + n] = pixels[i, n][0]

    return vector

# Function to calculate the difference between two image vectors
def vectorDifference(vector1, vector2):

    # Define variables
    difference = 0

    # Calculate difference
    for i in range(len(vector1)):
        difference = difference + abs(vector1[i] - vector2[i])

    return difference

# Function to recognise a face from the saved known faces
def recogniseFace(facesDirectory, inputFace, size, threshold):

    # Define variables
    result = "Unknown"

    # Generate dictionary of all known people and their images
    people = []
    faces = os.listdir(os.fsencode(facesDirectory))
    for i in range(len(faces)):
        face = os.listdir(os.fsencode(facesDirectory + os.fsdecode(faces[i])))
        person = {
            "name": os.fsdecode(faces[i]),
            "images": []
        }
        for n in range(len(face)):
            person["images"].append(facesDirectory + os.fsdecode(faces[i]) + "/" + os.fsdecode(face[n]))
        people.append(person)
    
    # Generate data matrix containing vectors of all known faces
    matrix = []
    for person in people:
        for imageDirectory in person["images"]:
            image = Image.open(imageDirectory)
            image = greyscale(image)
            image = image.resize((size, size))
            data = {
                "name": person["name"],
                "vector": generateVector(image)
            }
            matrix.append(data)

    # Return unknown if there are no known faces stored
    if (len(matrix) == 0):
        return result

    # Generate vector for input face
    inputFace = greyscale(inputFace)
    inputFace = inputFace.resize((size, size))
    inputVector = generateVector(inputFace)

    # Compare input face to all known faces
    minimum = {
        "name": matrix[0]["name"],
        "difference": vectorDifference(inputVector, matrix[0]["vector"])
    }
    for i in range(1, len(matrix)):
        difference = vectorDifference(inputVector, matrix[i]["vector"])
        if (difference < minimum["difference"]):
            minimum["name"] = matrix[i]["name"]
            minimum["difference"] = difference
    
    # Calculate result
    if (minimum["difference"] < threshold):
        result = minimum["name"]

    return result

# Main
#image = Image.open("Images/Input/facehard.jpg")
#image = detectFaces("Features/features.txt", image, 24, 24, 48, 8000)[0]

#image = Image.open("Images/Input/facesmall.jpg")
#image = image.rotate(180)
#image = detectFaces("Features/features.txt", image, 24, 24, 48, 7000)[0]

#image = Image.open("Images/Input/facepi2.jpg")
#image = detectFaces("Features/features.txt", image, 16, 16, 48, 4800)[0]

#image = Image.open("Images/Input/facepi2.jpg")
#image = resizeImage(image, 0.5)
#image = detectFaces("Features/features.txt", image, 24, 24, 200, 4000)[0]
#image.show()

#image = Image.open("Images/Input/facepi2.jpg")
#image = resizeImage(image, 0.5)
#data = detectFaces("Features/features.txt", image, 24, 24, 200, 4000)
#data[0].show()
#data[1][0].show()

#image = Image.open("Images/Input/facepi2.jpg")
#image = resizeImage(image, 0.5)
#face = detectFaces("Features/features.txt", image, 24, 24, 200, 4000)[1][0]
#print(recogniseFace("Images/Faces/", face, 100, 500000))

name = "Dan_Guerrero"
image = Image.open("Images/Input/testing/" + name + "/" + name + "_0001.jpg")
faces = detectFaces("Features/features.txt", image, 6, 6, 80, 10000)
faces[0].show()
