from flask import Flask, request, jsonify,  render_template
import tensorflow as tf
from Face_race_classification_with_FaceNet import face_fair_check

import os

app = Flask(__name__, template_folder='templates')
race_check = face_fair_check('facenet_model.h5')
# 이미지를 저장할 디렉토리 설정
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 허용되는 파일 확장자 설정
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html') # templates 폴더의 index.html 파일 렌더링


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    gender = int(request.form['gender'])
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # 파일의 안전한 이름 생성
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        # 파일 저장
        file.save(filename)       
        print(app.config['UPLOAD_FOLDER'] + file.filename) 
        img = race_check.get_face(app.config['UPLOAD_FOLDER'] + "/"+ file.filename)
        vector = race_check.get_embedded_face(img)
        result = race_check.get_face_race(vector, gender) #male 0, female 1W
        os.remove(app.config['UPLOAD_FOLDER'] + "/"+ file.filename)
        return  jsonify({'message' : result}), 200
    else:   
        return jsonify({'error': 'File type not allowed'}), 400



if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True) # 디렉토리 생성
    app.run(debug=True, host='0.0.0.0', port='80')

# if __name__ == '__main__':
#     os.makedirs(UPLOAD_FOLDER, exist_ok=True)  
#     app.run(debug=True)