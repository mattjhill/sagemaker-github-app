FROM public.ecr.aws/lambda/python:3.8

COPY model_data/config.json model_data/pytorch_model.bin ./model_data/
COPY app.py requirements.txt ./
RUN python3.8 -m pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html -t .
# Command can be overwritten by providing a different command in the template directly.
CMD ["app.lambda_handler"]
