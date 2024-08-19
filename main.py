from PIL import Image, ImageOps
import pandas as pd

image_path = '10000071750001.jpg'
# Load the image
image = Image.open(image_path)

# Converting the Image to Grayscale
gray_image = ImageOps.grayscale(image)

# Black pixel limit verification
dark_pixel_threshold = 200

# Convert the image to binary format
binary_image = gray_image.point(lambda x: 0 if x < 128 else 255, '1')

# Array to store the student number
student_number = []

# Quantity of alternatives
alternatives_quantity = 5
alternatives = ['A', 'B', 'C', 'D', 'E']

# Quantity of questions
questions_quantity = 30

# The x-coordinate of the first column and spacing between columns
column_start_x = 326
column_spacing = 47

# The y-coordinate of the first row and spacing between rows
row_start_y = 1613
row_spacing = 47

# The width and height of each circle
circle_width = 41
circle_height = 41

# Threshold value
black_pixels_threshold = 1000  # General threshold value set to 35

# Array to store the student number
student_answers = {}

# Check the density of the marked area for each column and row
for row_index in range(questions_quantity):  # For 30 questions
    row = row_start_y + row_index * row_spacing
    for col_index in range(alternatives_quantity):  # For 5 columns
        col = column_start_x + col_index * column_spacing
        # Crop the relevant area
        mark_area = binary_image.crop((col, row, col + circle_width, row + circle_height))
        # Count the number of black pixels
        black_pixels_count = sum(1 for pixel in mark_area.getdata() if pixel == 0)
        # If there are more black pixels than the threshold value, accept it as marked
        if black_pixels_count > black_pixels_threshold:
            student_answers[row_index + 1] = alternatives[col_index]
            break  # Move to the next column
    if not student_answers.get(row_index + 1):
        student_answers[row_index + 1] = '-'

print(student_answers)