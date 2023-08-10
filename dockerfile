FROM amazon/aws-lambda-python:3.8

RUN /var/lang/bin/python3.8 -m pip install --upgrade pip

RUN yum install git -y

RUN git clone --branch serverless https://github.com/YoonHyunWoo/FINDNA.git

RUN cp -r FINDNA/* ${LAMBDA_TASK_ROOT}

RUN pip install tensorflow-cpu --no-cache-dir

RUN pip install mediapipe --no-cache-dir

RUN pip install opencv-python --no-cache-dir

RUN pip install numpy --no-cache-dir

COPY facenet_model.h5 ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.handler"]
