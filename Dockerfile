# Use AWS base image for python 3.12
FROM public.ecr.aws/lambda/python:3.12

# Install build-essential compiler and tools
RUN microdnf update -y && microdnf install -y gcc-c++ make

# Copy requirements.txt file

COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install python dependencies

RUN pip install -r requirements.txt

# Copy function code
COPY travelAgent.py ${LAMBDA_TASK_ROOT}

# Set the permissions to make the file executable

RUN chmod +x travelAgent.py

# Set the CMD to your handler
CMD ["travelAgent.lambda_handler"]