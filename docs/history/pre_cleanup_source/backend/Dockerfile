FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r /tmp/requirements.txt
COPY backend/ backend/
COPY database/ database/
COPY rul_synthesis/ rul_synthesis/
COPY config.py config.py
COPY examples/ examples/
COPY data/runtime/dashboard/ frontend/dist/
COPY phase1_final/ phase1_final/
COPY phase2/current_target/ phase2/current_target/
COPY data/phase3_rul/scenarios.json data/phase3_rul/scenarios.json
ENV PCB_FRONTEND_BUILD=/app/frontend/dist PCB_DEVICE=cpu YOLO_CONFIG_DIR=/tmp/ultralytics MPLCONFIGDIR=/tmp/matplotlib
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
