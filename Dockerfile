FROM python:3.10    

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
ENV MAX_JOBS=1
RUN pip install torch transformers pandas tqdm scikit-learn datasets tensorflow numpy --progress-bar off


#CMD ['python3','/app/Model_output/model_output.py']

