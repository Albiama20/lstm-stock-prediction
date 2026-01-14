from pipelines.full_pipeline import full_pipeline

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

prediction_only = True

if __name__ == "__main__":
    full_pipeline(prediction_only)