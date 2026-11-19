# FIXME: Choose an appropriate base image
FROM python:3.10-slim

WORKDIR /app

# FIXME: Copy necessary files and install dependencies
# We need packages like torch (cpu-only to keep size smaller), torchvision, fastapi, uvicorn, nest_asyncio, and Pillow.
RUN pip install --no-cache-dir fastapi uvicorn Pillow nest_asyncio \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

# Copy our model weights and our application code
COPY model.pt /app/model.pt
# In a typical production setup, the Python application source would be copied as well
# COPY app.py /app/app.py

# EXPOSE the port FastAPI app is configured to run on
EXPOSE 8000

# FIXME: Set up the command to run your FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
