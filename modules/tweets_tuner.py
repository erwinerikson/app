"""Module tuner"""

import tensorflow as tf
import tensorflow_transform as tft
import keras_tuner as kt
from keras_tuner.engine import base_tuner
from keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs
from typing import NamedTuple, Dict, Text, Any  #pylint: disable=wrong-import-order

from modules.tweets_trainer import EMBEDDING_DIM, SEQUENCE_LENGTH, VOCAB_SIZE
from modules.tweets_transform import FEATURE_KEY, LABEL_KEY, transformed_name

def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern,
             tf_transform_output,
             num_epochs,
             batch_size=64)->tf.data.Dataset:
    """Get post_tranform feature & create batches of data"""
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )
    # create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key = transformed_name(LABEL_KEY))
    return dataset

vectorize_layer = layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH)

stop_early = tf.keras.callbacks.EarlyStopping(
    monitor='val_binary_accuracy',
    mode='max',
    verbose=1,
    patience=10
)

def model_builder(hp):
    """Build machine learning model"""
    inputs = tf.keras.Input(shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)
    reshaped_narrative = tf.reshape(inputs, [-1])
    x = vectorize_layer(reshaped_narrative)
    x = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, name="embedding")(x)
    x = layers.GlobalAveragePooling1D()(x)
    hp_units1 = hp.Int('units1', min_value=32, max_value=512, step=32)
    x = layers.Dense(units=hp_units1, activation='relu')(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs = outputs)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(
        loss = 'binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )
    return model


TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                             ('fit_kwargs', Dict[Text, Any])])

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Build the tuner using the KerasTuner API.
    Args:
    fn_args: Holds args used to tune models as name/value pairs.

    Returns:
    A namedtuple contains the following:
      - tuner: A BaseTuner that will be used for tuning.
      - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                    model , e.g., the training and validation dataset. Required
                    args depend on the above tuner's implementation.
    """
    # Memuat training dan validation dataset yang telah di-preprocessing
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_set = input_fn(fn_args.train_files, tf_transform_output, 10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, 10)
    vectorize_layer.adapt(
        [j[0].numpy()[0] for j in [
            i[0][transformed_name(FEATURE_KEY)]
                for i in list(train_set)]])

    # Mendefinisikan strategi hyperparameter tuning
    tuner = kt.RandomSearch(
        hypermodel = lambda hp: model_builder(kt.HyperParameters()),
        objective = kt.Objective('val_binary_accuracy', direction='max'),
        max_trials = 3,
        overwrite = True,
        directory = fn_args.working_dir,
        project_name = "kt_RandomSearch"
    )
    tuner.search(
        train_set,
        epochs=5,
        validation_data=val_set
    )
    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "callbacks":[stop_early],
            'x': train_set,
            'validation_data': val_set,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps
        }
    )
