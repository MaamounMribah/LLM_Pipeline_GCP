FROM python:3.10

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY /evaluate_fine_tuned_model /app
COPY /Fine_tuned_model_output /app
COPY /Model_output /app 
COPY /Preprocess_data /app

# Install any needed packages specified in requirements.txt
RUN pip install torch transformers pandas tqdm scikit-learn datasets tensorflow numpy



