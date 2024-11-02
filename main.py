import os
from PIL import Image, ImageOps
import pandas as pd
import glob
from pathlib import Path
from datetime import datetime
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from dotenv import load_dotenv
# import easyocr
import pytesseract
from pdf2image import convert_from_path  # Add this for PDF conversion

load_dotenv(override=True)

# EASYOCR APPROACH
# ocr_reader = easyocr.Reader(['pt'])

base_path = os.getenv("base_path")
images_folder_path = f"{base_path}/imagens"
template_path = f"{base_path}/gabarito.xlsx"

images_list = glob.glob(f"{images_folder_path}/*")

results = []
df = pd.read_excel(template_path)
template_dict = df.set_index('Questão')['Resposta'].to_dict()

# Quantity of questions
question_blocks_quantity = int(os.getenv("question_blocks_quantity"))
questions_per_block_quantity = int(os.getenv("questions_per_block_quantity"))
total_questions = int(os.getenv("total_questions"))
question_index_initial_value = int(os.getenv("question_index_initial_value"))
black_pixels_threshold = int(os.getenv("black_pixels_threshold"))

results.append({
    'name': 'Gabarito',
    'file': 'N/A',
    'correct_questions_quantity': total_questions,
    'correct_questions_percentage': 100,
    'answers': template_dict
})

for file_path in images_list:
    # Check if file is a PDF
    if file_path.lower().endswith('.pdf'):
        # Convert PDF to images
        pages = convert_from_path(file_path)
        image = pages[0]  # Assuming the first page is the one we need
    else:
        # Load the image directly if it's not a PDF
        image = Image.open(file_path)

    print(f"Processing {file_path}")

    # Converting the Image to Grayscale
    gray_image = ImageOps.grayscale(image)

    # Getting name
    name_start_x = int(os.getenv("name_start_x"))
    name_start_y = int(os.getenv("name_start_y"))
    name_height = int(os.getenv("name_height"))
    name_width = int(os.getenv("name_width"))
    name_area = gray_image.crop((name_start_x, name_start_y, name_start_x + name_width, name_start_y + name_height))

    #### FOR TESTING PURPOSES ONLY ####
    # name_area.save('name_area.png')
    # exit()
    #### FOR TESTING PURPOSES ONLY ####

    name = pytesseract.image_to_string(name_area)

    # EASYOCR APPROACH
    # img_byte_arr = io.BytesIO()
    # name_area.save(img_byte_arr, format='PNG')
    # img_byte_arr = img_byte_arr.getvalue()
    # ocr_result = ocr_reader.readtext(img_byte_arr)
    
    # name = ''
    # for result_set in ocr_result:
    #     for item in result_set:
    #         if (isinstance(item, str)) and len(item) > 5:
    #             name = item

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

    # Array to store the student number
    student_answers = {}
    question_index = question_index_initial_value - 1
    correct_questions_quantity = 0

    # Check the density of the marked area for each column and row
    for block_index in range(question_blocks_quantity):  # For 4 blocks
        for row_index in range(questions_per_block_quantity):  # For 30 questions
            row = row_start_y + row_index * row_spacing + row_index * circle_height
            for col_index in range(alternatives_quantity):  # For 5 columns
                col = column_start_x + \
                    col_index * column_spacing + \
                    col_index * circle_width + \
                    block_index * block_spacing + \
                    block_index * alternatives_quantity * circle_width + \
                    block_index * (alternatives_quantity - 1) * column_spacing
                # Crop the relevant area
                mark_area = binary_image.crop((col, row, col + circle_width, row + circle_height))
                
                #### FOR TESTING PURPOSES ONLY ####
                # if question_index + 1 == 81:
                #     question_area = gray_image.crop((col, row, col + circle_width, row + circle_height))
                #     question_area.save(f"question_{question_index+1}_item_{col_index}.png")
                #### FOR TESTING PURPOSES ONLY ####
                
                # Count the number of black pixels
                black_pixels_count = sum(1 for pixel in mark_area.getdata() if pixel == 0)
                # If there are more black pixels than the threshold value, accept it as marked
                if black_pixels_count > black_pixels_threshold:
                    student_answers[question_index + 1] = alternatives[col_index]
            if not student_answers.get(question_index + 1):
                student_answers[question_index + 1] = '-'
            if student_answers.get(question_index + 1) == template_dict.get(question_index + 1):
                correct_questions_quantity += 1
            question_index += 1
            if question_index + 1 > question_index_initial_value + total_questions:
                break
        if question_index + 1 > question_index_initial_value + total_questions:
            break
    #### FOR TESTING PURPOSES ONLY ####
    # exit()
    #### FOR TESTING PURPOSES ONLY ####

    results.append({
        'name': name,
        'file': Path(file_path).stem,
        'correct_questions_quantity': correct_questions_quantity,
        'correct_questions_percentage': (correct_questions_quantity / total_questions) * 100,
        'answers': student_answers
    })

file_name = base_path + '/resultados_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.xlsx'
workbook = Workbook()

worksheet = workbook.active
worksheet.title = "Gabarito e resultados"

spreadsheet_row_index = 1
worksheet.cell(row=spreadsheet_row_index, column=1).value = 'Arquivo'
worksheet.column_dimensions['A'].width = 50
worksheet.cell(row=spreadsheet_row_index, column=2).value = 'Nome'
worksheet.column_dimensions['B'].width = 50
worksheet.cell(row=spreadsheet_row_index, column=3).value = 'Questões corretas'
worksheet.column_dimensions['C'].width = 20
worksheet.cell(row=spreadsheet_row_index, column=4).value = 'Percentual de acerto'
worksheet.column_dimensions['D'].width = 20
for column_index in range(total_questions):
    worksheet.column_dimensions[get_column_letter(column_index + 5)].width = 5
    worksheet.cell(row=spreadsheet_row_index, column=column_index + 5).value = question_index_initial_value + column_index
spreadsheet_row_index += 1
for result in results:
    worksheet.cell(row=spreadsheet_row_index, column=1).value = result['file']
    worksheet.cell(row=spreadsheet_row_index, column=2).value = result['name']
    worksheet.cell(row=spreadsheet_row_index, column=3).value = result['correct_questions_quantity']
    worksheet.cell(row=spreadsheet_row_index, column=4).value = result['correct_questions_percentage']
    for column_index in range(total_questions):
        worksheet.cell(row=spreadsheet_row_index, column=column_index + 5).value = result['answers'][question_index_initial_value + column_index]
    spreadsheet_row_index += 1

workbook.save(file_name)
workbook.close()
