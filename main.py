import os
from PIL import Image, ImageOps
import pandas as pd
import glob
from pathlib import Path
from datetime import datetime
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from dotenv import load_dotenv

load_dotenv(override=True)

base_path = os.getenv("base_path")
images_folder_path = f"{base_path}/imagens"
template_path = f"{base_path}/gabarito.xlsx"

images_list = glob.glob(f"{images_folder_path}/*")

results = []
df = pd.read_excel(template_path)
template_dict = df.set_index('Questão')['Resposta'].to_dict()

# Quantity of questions
question_blocks_quantity = int(os.getenv("question_blocks_quantity"))

# Quantity of questions
questions_per_block_quantity = int(os.getenv("questions_per_block_quantity"))

total_questions = question_blocks_quantity * questions_per_block_quantity

results.append({
    'name': 'Gabarito',
    'correct_questions_quantity': total_questions,
    'correct_questions_percentage' : 100,
    'answers': template_dict
})

for image_path in images_list:
    # Load the image
    image = Image.open(image_path)

    # Converting the Image to Grayscale
    gray_image = ImageOps.grayscale(image)

    # Black pixel limit verification
    dark_pixel_threshold = int(os.getenv("dark_pixel_threshold"))

    # Convert the image to binary format
    binary_image = gray_image.point(lambda x: 0 if x < 128 else 255, '1')

    # Array to store the student number
    student_number = []

    # Quantity of alternatives
    alternatives_quantity = 5
    alternatives = ['A', 'B', 'C', 'D', 'E']

    # The x-coordinate of the first column and spacing between columns
    column_start_x = int(os.getenv("column_start_x"))
    column_spacing = int(os.getenv("column_spacing"))
    block_spacing = int(os.getenv("block_spacing"))

    # The y-coordinate of the first row and spacing between rows
    row_start_y = int(os.getenv("row_start_y"))
    row_spacing = int(os.getenv("row_spacing"))

    # The width and height of each circle
    circle_width = int(os.getenv("circle_width"))
    circle_height = int(os.getenv("circle_height"))

    # Threshold value
    black_pixels_threshold = 1000  # General threshold value set to 35

    # Array to store the student number
    student_answers = {}
    question_index_initial_value = int(os.getenv("question_index_initial_value"))
    question_index = question_index_initial_value - 1
    correct_questions_quantity = 0

    # Check the density of the marked area for each column and row
    for block_index in range(question_blocks_quantity):  # For 4 blocks
        for row_index in range(questions_per_block_quantity):  # For 30 questions
            row = row_start_y + row_index * row_spacing
            for col_index in range(alternatives_quantity):  # For 5 columns
                col = column_start_x + col_index * column_spacing + block_index * block_spacing
                # Crop the relevant area
                mark_area = binary_image.crop((col, row, col + circle_width, row + circle_height))
                # Count the number of black pixels
                black_pixels_count = sum(1 for pixel in mark_area.getdata() if pixel == 0)
                # If there are more black pixels than the threshold value, accept it as marked
                if black_pixels_count > black_pixels_threshold:
                    student_answers[question_index + 1] = alternatives[col_index]
                    break  # Move to the next column
            if not student_answers.get(question_index + 1):
                student_answers[question_index + 1] = '-'
            if student_answers.get(question_index + 1) == template_dict.get(question_index + 1):
                correct_questions_quantity += 1
            question_index += 1

    results.append({
        'name': Path(image_path).stem,
        'correct_questions_quantity': correct_questions_quantity,
        'correct_questions_percentage': (correct_questions_quantity / total_questions) * 100,
        'answers': student_answers
    })

file_name = base_path + '/resultados_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.xlsx'
workbook = Workbook()

worksheet = workbook.active
worksheet.title = "Gabarito e resultados"

spreadsheet_row_index = 1
worksheet.cell(row=spreadsheet_row_index, column=1).value = ''
worksheet.column_dimensions['A'].width = 50
worksheet.cell(row=spreadsheet_row_index, column=2).value = 'Questões corretas'
worksheet.column_dimensions['B'].width = 20
worksheet.cell(row=spreadsheet_row_index, column=3).value = 'Percentual de acerto'
worksheet.column_dimensions['C'].width = 20
for column_index in range(total_questions):
    worksheet.column_dimensions[get_column_letter(column_index + 4)].width = 5
    worksheet.cell(row=spreadsheet_row_index, column=column_index + 4).value = question_index_initial_value + column_index

spreadsheet_row_index += 1
for result in results:
    worksheet.cell(row=spreadsheet_row_index, column=1).value = result['name']
    worksheet.cell(row=spreadsheet_row_index, column=2).value = result['correct_questions_quantity']
    worksheet.cell(row=spreadsheet_row_index, column=3).value = result['correct_questions_percentage']
    for column_index in range(total_questions):
        worksheet.cell(row=spreadsheet_row_index, column=column_index + 4).value = result['answers'][question_index_initial_value + column_index]
    spreadsheet_row_index += 1

workbook.save(file_name)
workbook.close()