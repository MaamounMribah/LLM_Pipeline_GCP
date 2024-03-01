FROM python:3.10

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY evaluate_fine_tuned_model /app/evaluate_fine_tuned_model
COPY Fine_tuned_model_output /app/Fine_tuned_model_output
COPY Model_output /app/Model_output
COPY Preprocess_data /app/Preprocess_data

# Install any needed packages specified in requirements.txt
RUN pip install torch transformers pandas tqdm scikit-learn datasets tensorflow numpy

#RUN pip install torch transformers

#CMD ['python3','/app/Model_output/model_output.py']

