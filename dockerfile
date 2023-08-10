FROM amazon/aws-lambda-python:3.8

RUN /var/lang/bin/python3.8 -m pip install --upgrade pip

RUN yum install git -y

RUN git clone --branch serverless https://github.com/YoonHyunWoo/FINDNA.git

RUN cp -r FINDNA/* /var/task

RUN pip install tensorflow

RUN pip install mediapipe

RUN install cv2

RUN pip install numpy

COPY facenet_model.h5 /var/task

CMD ["lambda_function.lambda_handler"]
