from Face_race_classification_with_FaceNet import face_fair_check
import os
import io
from PIL import Image

race_check = face_fair_check('/var/task/facenet_model.h5')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def lambda_handler(event, context):
    file_info = event['file']
    gender = int(event['gender'])
    filename = file_info['name']
    
    if not allowed_file(filename):
        return {'statusCode': 400, 'body': 'File type not allowed'}

    file_content = file_info['content']
    binary_file = io.BytesIO(file_content)

    img = race_check.get_face(binary_file)
    vector = race_check.get_embedded_face(img)
    result = race_check.get_face_race(vector, gender) #male 0, female 1W

    return {
        'statusCode': 200,
        'body': {'message': result}
    }
