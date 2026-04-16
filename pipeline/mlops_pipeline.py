import kfp
from kfp import dsl
from kfp.dsl import ContainerSpec
from kfp import kubernetes


@dsl.container_component
def data_ingestion_op():
    return ContainerSpec(
        image='atharvarai07/my-mlops-app:latest',
        command=['python', 'src/data_ingestion.py'],
        args=[]
    )


@dsl.container_component
def data_preprocessing_op():
    return ContainerSpec(
        image='atharvarai07/my-mlops-app:latest',
        command=['python', 'src/data_processing.py'],
        args=[]
    )


@dsl.container_component
def model_training_op():
    return ContainerSpec(
        image='atharvarai07/my-mlops-app:latest',
        command=['python', 'src/model_trainer.py'],
        args=[]
    )


@dsl.pipeline(
    name='MLOps Pipeline',
    description='A pipeline that performs data ingestion, preprocessing, and model training.'
)
def mlops_pipeline():
    data_ingestion = data_ingestion_op()
    kubernetes.set_image_pull_policy(data_ingestion, 'Never')

    data_preprocessing = data_preprocessing_op().after(data_ingestion)
    kubernetes.set_image_pull_policy(data_preprocessing, 'Never')

    model_training = model_training_op().after(data_preprocessing)
    kubernetes.set_image_pull_policy(model_training, 'Never')


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(
        mlops_pipeline,
        'mlops_pipeline.yaml'
    )
