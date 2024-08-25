from PIL import Image, ImageOps
import pandas as pd
import glob
from pathlib import Path
from datetime import datetime
from openpyxl import Workbook
from openpyxl.styles import Alignment

def fix_column_width(worksheet):
    dims = {}
    for row in worksheet.rows:
        for cell in row:
            if cell.value:
                dims[cell.column_letter] = max((dims.get(cell.column_letter, 0), len(str(cell.value))))
    for col, value in dims.items():
        worksheet.column_dimensions[col].width = value

base_path = '/Users/tassiovalle/Documents/optical-mark-recognizer'
images_folder_path = f"{base_path}/images"
template_path = f"{base_path}/templates/gabarito.xlsx"

images_list = glob.glob(f"{images_folder_path}/*")

results = []
df = pd.read_excel(template_path)
template_dict = df.set_index('Questão')['Resposta'].to_dict()

# Quantity of questions
question_blocks_quantity = 4

# Quantity of questions
questions_per_block_quantity = 30

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
    dark_pixel_threshold = 200

    # Convert the image to binary format
    binary_image = gray_image.point(lambda x: 0 if x < 128 else 255, '1')

    # Array to store the student number
    student_number = []

    # Quantity of alternatives
    alternatives_quantity = 5
    alternatives = ['A', 'B', 'C', 'D', 'E']

    # The x-coordinate of the first column and spacing between columns
    column_start_x = 326
    column_spacing = 47
    block_spacing = 570

    # The y-coordinate of the first row and spacing between rows
    row_start_y = 1613
    row_spacing = 48

    # The width and height of each circle
    circle_width = 41
    circle_height = 41

    # Threshold value
    black_pixels_threshold = 1000  # General threshold value set to 35

    # Array to store the student number
    student_answers = {}
    question_index = 0
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

file_name = 'resultados_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.xlsx'
workbook = Workbook()

worksheet = workbook.active
worksheet.title = "Gabarito e resultados"

spreadsheet_column_index = 1
worksheet.cell(row=1, column=spreadsheet_column_index).value = ''
for row_index in range(1, total_questions + 1):
    worksheet.cell(row=row_index+1, column=spreadsheet_column_index).value = row_index
worksheet.cell(row=total_questions + 2, column=spreadsheet_column_index).value = 'Questões corretas'
worksheet.cell(row=total_questions + 3, column=spreadsheet_column_index).value = 'Percentual de acerto'

spreadsheet_column_index += 1
for result in results:
    worksheet.cell(row=1, column=spreadsheet_column_index).value = result['name']
    for row_index in range(1, total_questions + 1):
        worksheet.cell(row=row_index+1, column=spreadsheet_column_index).value = result['answers'][row_index]
    worksheet.cell(row=total_questions + 2, column=spreadsheet_column_index).value = result['correct_questions_quantity']
    worksheet.cell(row=total_questions + 3, column=spreadsheet_column_index).value = result['correct_questions_percentage']
    spreadsheet_column_index += 1

fix_column_width(worksheet)
workbook.save(file_name)
workbook.close()