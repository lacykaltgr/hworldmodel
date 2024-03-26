FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /app
COPY scripts/start_jupyter.sh /

RUN chmod +x ../start_jupyter.sh
CMD ["../start_jupyter.sh"]
ENV PYTHONPATH="${PYTHONPATH}:/app"
