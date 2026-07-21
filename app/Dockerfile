FROM python:3.10-slim

WORKDIR /app

# Sin esto, pip a veces no encuentra ruedas para pandas==1.3.5 en Python 3.10.
RUN pip install --no-cache-dir --upgrade pip

# PyTorch CPU (misma versión que el TFM, 2.1.2, pero build CPU en vez de cu121).
RUN pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Extensiones de PyG en su variante CPU, mismas versiones que requirements.txt
# del TFM. Se fija el índice exacto porque estas ruedas son sensibles a la
# combinación torch+cpu/cuda y no están en PyPI normal.
RUN pip install --no-cache-dir \
    torch_geometric==2.3.1 \
    torch-scatter==2.1.2 \
    torch-sparse==0.6.18 \
    torch-cluster==1.6.3 \
    torch-spline-conv==1.2.2 \
    -f https://data.pyg.org/whl/torch-2.1.2+cpu.html

RUN pip install --no-cache-dir torch_geometric_temporal==0.54.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.py data_utils.py api.py ./
COPY model_data/ ./model_data/
COPY static/ ./static/

EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
