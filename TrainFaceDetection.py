
# Imported libraries
from PIL import Image
import os
import operator
import time
import math as maths

# Function to generate a feature
def generateFeature(type, inverted, start, size):

    features = ["horizontal_edge", "vertical_edge", "horizontal_line", "vertical_line", "square"]
    booleans = [False, True]

    feature = {
        "type": features[type],
        "inverted": booleans[inverted],
        "start": start,
        "size": size,
        "score": 0
    }

    return feature

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

# Function to generate integral images for the datasets
def generateDatasetIntegral(datasetDirectory):

    # Define variables
    datasetIntegrals = []

    # Generate list of dataset image directories
    dataset = os.listdir(os.fsencode(datasetDirectory))
    for i in range(len(dataset)):
        dataset[i] = datasetDirectory + os.fsdecode(dataset[i])

    # Generate integral for each image in the dataset
    for image in dataset:
        datasetIntegrals.append(generateIntegral(greyscale(Image.open(image)).resize((24, 24))))

    return datasetIntegrals


# Function to score a feature
def scoreFeature(feature, faceIntegrals, backgroundIntegrals):

    # Define variables
    score = 0

    # Iterate through each integral image
    for face in faceIntegrals:
        result = applyFeature(feature, face)
        if (result > 0):
            score = score + 1
        else:
            score = score - 1

    for background in backgroundIntegrals:
        result = applyFeature(feature, background)
        if (result > 0):
            score = score - 1
        else:
            score = score + 1

    return score

# Function to generate a list of all possible features
def generateFeatures():

    # Define variables
    features = []
    intervals = [[2, 1], [1, 2], [3, 1], [1, 3], [2, 2]]
    faceIntegrals = generateDatasetIntegral("Datasets/Face Detection/Faces/")
    backgroundIntegrals = generateDatasetIntegral("Datasets/Face Detection/Backgrounds/")

    # Generate and score all features
    for type in range(5):
        os.system("cls")
        print("\n" + str(type * 20) + "%\n")
        for inverted in range(2):
            for startX in range(24):
                for startY in range(24):
                    for sizeX in range(intervals[type][0], 23 - startX, intervals[type][0]):
                        for sizeY in range(intervals[type][1], 23 - startY, intervals[type][1]):
                            feature = generateFeature(type, inverted, [startX, startY], [sizeX, sizeY])
                            feature["score"] = scoreFeature(feature, faceIntegrals, backgroundIntegrals)
                            features.append(feature)

    return features

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
def applyFeature(feature, integral):

    # Define variables
    result = 0
    start = [feature["start"][0], feature["start"][1]]
    size = [feature["size"][0], feature["size"][1]]

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

# Main
def main():

    # Get user input
    os.system("cls")
    output = input("\nOutput file name: ")
    os.system("cls")

    # Generate a list of features
    start = time.time()
    features = generateFeatures()

    # Sort features and remove duplicates
    features.sort(reverse=True, key=operator.itemgetter("score"))
    temp = []
    for feature in features:
        temp.append(str(feature["type"]) + "|" + str(feature["inverted"]) + "|" + str(feature["start"]) + "|" + str(feature["size"]) + "|" + str(feature["score"]))
    features = list(dict.fromkeys(temp))

    # Write to file
    f = open("Features/" + output + ".txt", "a")
    for feature in features:
        f.write(feature + "\n")
    f.close()
    os.system("cls")
    print("\nWritten features to file '" + output + "' successfully.")
    print("Training took " + str(int(time.time() - start)) + " seconds.\n")

    return

main()
