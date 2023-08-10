from Face_race_classification_with_FaceNet import face_fair_check
import os
import io
from PIL import Image
import base64
import json
import pyheif

race_check = face_fair_check('/var/task/facenet_model.h5')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'heic'}
UPLOAD_FOLDER = '/tmp'

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_image(file_content, filename):
    if filename.lower().endswith('.heic'):
        heif_file = pyheif.read(file_content)
        image = Image.frombytes(
            heif_file.mode, 
            heif_file.size, 
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )
        return image
    else:
        return Image.open(file_content)

def handler(event, context):
    event = event['body-json']
    file_info = event['file']
    gender = int(event['gender'])
    filename = file_info['name']

    if not allowed_file(filename):
        return {'statusCode': 400, 'body': 'File type not allowed'}

    # /tmp 디렉토리에 파일 저장
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, 'wb') as file:
        file.write(base64.b64decode(file_info['content']))

    # 이미지 처리
    img = race_check.get_face(filepath)
    vector = race_check.get_embedded_face(img)
    result = race_check.get_face_race(vector, gender) # male 0, female 1

    # 임시 파일 제거
    os.remove(filepath)

    return {
        'statusCode': 200,
        'body': json.dumps({'message': result})
    }

