FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime
RUN python -m pip install --no-cache-dir boto3==1.40.31 Pillow==11.3.0
RUN python -c "import torch, torchvision, boto3, PIL; print(torch.__version__, torchvision.__version__)"
WORKDIR /lab
COPY train.py fruit_classes.py infer.py /lab/
CMD ["python", "-u", "/lab/train.py"]
