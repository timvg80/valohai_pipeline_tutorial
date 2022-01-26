from valohai import Pipeline

def main(config) -> Pipeline:

    #Create a pipeline called "utilspipeline".
    pipe = Pipeline(name="utilspipeline", config=config)

    # Define the pipeline nodes.
    preprocess = pipe.execution("preprocess-dataset")
    train = pipe.execution("train-model")

    # Configure the pipeline, i.e. define the edges.
    preprocess.output("preprocessed_mnist.npz").to(train.input("dataset"))

    return pipe
