import cv2

def process_image(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 
        11, 
        4
    )
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume the answer regions are detected correctly (this part can vary)
    answer_key = 'ABCDE'
    detected_answers = []
    
    for cnt in contours:
        # Approximate the contour
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        
        # Assume we have rectangular regions for the answer
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Extract the region of interest (ROI)
            roi = thresh[y:y+h, x:x+w]
            
            # Split the ROI into columns corresponding to choices A-E
            cell_width = w // 5
            marks = []
            
            for i in range(5):
                cell = roi[:, i*cell_width:(i+1)*cell_width]
                
                # Count the number of white pixels in the cell
                white_pixel_count = cv2.countNonZero(cell)
                
                # If there are enough white pixels, consider it as marked
                if white_pixel_count > cell.size * 0.5:  # 50% threshold
                    marks.append(answer_key[i])
                
            detected_answers.append(marks)
    
    return detected_answers

if __name__ == "__main__":
    image_path = '10000071750001.jpg'  # Provide the path to your answer sheet image
    answers = process_image(image_path)
    print(answers)