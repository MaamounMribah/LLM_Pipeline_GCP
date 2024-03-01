import kfp
from kfp import dsl
from kfp import Client

import os
import random
import subprocess
import string

def random_suffix() -> string:
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))

def run_command(command):
    """
    Runs a command using subprocess.run and captures the output and errors.
    """
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Command succeeded: {result.stdout}")
    else:
        print(f"Command failed with error: {result.stderr}")

def build_images():
    build_commands = [
        "DOCKER_BUILDKIT=1 docker build -t maamounm/llm_pipeline:latest .",
    ]
    for command in build_commands:
        run_command(command)

def push_images():
    push_commands = [
        "docker push maamounm/llm_pipeline:latest",
    ]
    for command in push_commands:
        run_command(command)



def preprocess_data_op(dataset : str, split: str) :
    return dsl.ContainerOp(
        name="Data Preprocessing",
        image='maamounm/llm_pipeline:latest',
        command=['python3','/app/Preprocess_data/preprocess_data.py'],
        arguments=[
            '--dataset', dataset,
            '--split', split,
        ],
    )




def bert_output_before_fine_tuning_op():
    return dsl.ContainerOp(
        name="output before fine tuning",
        image='maamounm/llm_pipeline:latest',
        command=['python3','/app/Model_output/model_output.py'],
        #arguments=[model_path,test_data],
    )

def bert_fine_tuned_model_output_op():
    
    return dsl.ContainerOp(
        name="output after fine tuning",
        image='maamounm/llm_pipeline:latest',
        command=['python3','/app/Fine_tuned_model_output/fine_tuned_model_output.py'],
        #arguments=[model_path,test_data],
        #resource_requests={"cpu": "4", "memory": "4Gi"},
        #resource_limits={"cpu": "8", "memory": "8Gi"},
    )

def evaluate_fine_tuned_model_op():
    #test_data='preprocessed_data.pkl'
    #model_path='model'
    return dsl.ContainerOp(
        name="fine tuned model evaluation",
        image='maamounm/llm_pipeline:latest',
        command=['python3','/app/evaluate_fine_tuned_model/evaluate_fine_tuned_model.py'],
        #arguments=[model_path,test_data],
    )

@dsl.pipeline(
    name='bert Fine-Tuning Pipeline',
    description='A pipeline that fine-tunes bert model with sprecific data.'
)

def llm_pipeline():
    dataset='ag_news'
    split='train[:1%]'
    preprocessed_data_path='preprocessed_data.pkl'
    test_data='preprocessed_data.pkl'
    model_path='gpt2_finetuned'
    input_data='test output text generated by gpt2'
    

    preprocess_task = preprocess_data_op(dataset=dataset, split=split)

    generate_output_task = bert_output_before_fine_tuning_op().after(preprocess_task)
    generate_output_after_fine_tuning_task=bert_fine_tuned_model_output_op().after(preprocess_task)
    evaluate_fine_tuned_model_task=evaluate_fine_tuned_model_op().after(generate_output_after_fine_tuning_task)
    



#endpoint="http://localhost:8080/"
endpoint="https://e972-41-227-19-17.ngrok-free.app/#/pipelines"
kfp_client= Client(host=endpoint)



pipeline_package_path='testing.yaml'
pipeline_name = "BERT Fine-Tuning Pipeline "+random_suffix()
pipeline_description = "A pipeline that fine-tunes BERT model with specific data."

experiment_name = "BERT Fine-Tuning Experiments"
experiment_description = "Experiments for fine-tuning BERT models"

if __name__ == "__main__":
    
    #build_images()
    #push_images()
    

    # Compile the pipeline to YAML
    kfp.compiler.Compiler().compile(pipeline_func=llm_pipeline, package_path='LLM_pipeline.yaml')

    # Define and create an experiment
    experiment_response = kfp_client.create_experiment(name=experiment_name, description=experiment_description)

    # Upload the pipeline
    pipeline_response = kfp_client.upload_pipeline('LLM_pipeline.yaml', pipeline_name=pipeline_name, description=pipeline_description)

    # Extract the experiment ID
    experiment_id = experiment_response.id


    # Extract the pipeline ID
    pipeline_id = pipeline_response.id

    # List versions for the uploaded pipeline and select the most recent version ID
    #version_id=kfp_client.list_pipeline_versions(pipeline_id=pipeline_id)

    
    # Create a run within the defined experiment using the uploaded pipeline and its version
    run_name = f"{pipeline_name} Run "
    #run = kfp_client.create_run_from_pipeline_func(llm_pipeline,arguments={})
    run_response = kfp_client.run_pipeline(experiment_id=experiment_id, job_name=run_name, pipeline_id=pipeline_id, params={})