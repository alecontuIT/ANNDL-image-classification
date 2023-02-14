import os
import shutil as shtl
from sklearn.model_selection import train_test_split
import custom_metrics
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
tfk = tf.keras
tfkl = tfk.layers
seed = 17560

def build_train_valid_dirs(all_data_dir_rel_path, train_dir_rel_path, valid_dir_rel_path,
                           valid_percentage=0.15):
    '''
    This method does a copy of data, but splitted in training and validation folders, according
    to the percentage of data that is passed to this method as parameter
    all_data_dir_rel_path,  |
    train_dir_rel_path,     |----> relative paths to the respective folders, starelative starting from pwd
    valid_dir_rel_path,     |

    valid_percentage is the % of data that must be in the validation directoy
    '''
    # building paths
    train_dir_path = os.path.join('.', train_dir_rel_path)
    valid_dir_path = os.path.join('.', valid_dir_rel_path)
    all_data_dir_path = os.path.join('.', all_data_dir_rel_path)

    # cleaning directories if already present
    if (os.path.isdir(train_dir_path)):
        shtl.rmtree(train_dir_path)
    if (os.path.isdir(valid_dir_path)):
        shtl.rmtree(valid_dir_path)
    
    # making training and validation directories
    os.mkdir(train_dir_rel_path)
    os.mkdir(valid_dir_rel_path)

    # for each subdir in all_data_dir_path, make one with the same name in train_dir_path and in
    # valid_dir_path and split the data in each all_data_dir_path subfolder, copying them into the
    # train_dir_path and valid_dir_path
    # subfolders according to the valid_percentage
    for dir in os.listdir(all_data_dir_rel_path):
        all_data_dir = os.path.join(all_data_dir_path, dir)
        if (os.path.isdir(all_data_dir)):
            valid_dir = os.path.join(valid_dir_path, dir)
            train_dir = os.path.join(train_dir_path, dir)
            os.mkdir(train_dir)
            os.mkdir(valid_dir)

        # pick all the files in the dir
        files = []
        for f in os.listdir(all_data_dir):
            files.append(f)

        # split in train and validation sets
        train, validation = train_test_split(files, test_size=valid_percentage, random_state=seed)
        for t in train:
            shtl.copyfile(os.path.join(all_data_dir, t), os.path.join(train_dir, t))
        for v in validation:
            shtl.copyfile(os.path.join(all_data_dir, v), os.path.join(valid_dir, v))



def count_images_in(path_to_dir):
    """
    This method return the number of images inside the folder path_to_dir passed as input
    """
    target_dir = os.path.join('.', path_to_dir)
    count = 0
    if (os.path.isdir(target_dir)):
        for class_dir in os.listdir(target_dir):
            class_dir = os.path.join(target_dir, class_dir)
            if (os.path.isdir(class_dir)):
                count += len(os.listdir(class_dir))
    return count
    
    
    
def callbacks(which_monitor, maxOrMin, patience, modelName):
    """
    which_monitor -> string to identify monitor metric in EarlyStopping
    maxOrMin      -> string to identify if in EarlyStopping we have to look at the max or at the min of
    the monitor before stopping
    patience      -> how much we have to wait, once it's reached the max or min to stop
    
    return        -> an array of callbacks (Early stopping and checkpoints of the best and of the last)
    """
    callbacks = []

    # Model checkpoint -> automatically save the model during training
    ckpt_dir = os.path.join('.', 'ckpts-' + modelName)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp_last.ckpt'),
                                                     save_weights_only=False, 
                                                     save_best_only=False)  
    callbacks.append(ckpt_callback)
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp_best.ckpt'),
                                                     save_weights_only=False,
                                                     save_best_only=True)

    callbacks.append(ckpt_callback)
    
    # Visualize Learning on Tensorboard
    tb_dir = os.path.join('.', 'tb_logs-' + modelName)
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
      
    # By default shows losses and metrics for both training and validation
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
                                               profile_batch=0,   
                                               histogram_freq=1)  
    callbacks.append(tb_callback)

    # Early Stopping
    es_callback = tf.keras.callbacks.EarlyStopping(monitor=which_monitor, mode=maxOrMin,
                                                   patience=patience, restore_best_weights=True)
    callbacks.append(es_callback)

    return callbacks
    


def metrics():
    '''
    returns an array of metrics usefull to evaluate a model during and after training
    '''
    _metrics = [
      "accuracy",
      custom_metrics.precision_m,
      custom_metrics.recall_m,
      custom_metrics.f1_m,
      tfk.metrics.CategoricalAccuracy(name="cat_acc"),
      tfk.metrics.TruePositives(name='tp'),
      tfk.metrics.FalsePositives(name='fp'),
      tfk.metrics.TrueNegatives(name='tn'),
      tfk.metrics.FalseNegatives(name='fn'),
      tfk.metrics.Precision(name='precision'),
      tfk.metrics.Recall(name='recall'),
      tfk.metrics.AUC(name='auc'),
      tfk.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]
    return _metrics
    
    

def build_tl_vgg_model(hiddens, neurons, input_shape, seed=seed):
    supernet = tfk.applications.VGG16(include_top=False, weights="imagenet", input_shape=input_shape)

    # Using the supernet as feature extractor
    supernet.trainable = False
    inputs = tfk.Input(shape=input_shape)
    x = supernet(inputs)
    x = tfkl.Flatten(name='Flattening')(x)
    
    # classifier
    if(hiddens == len(neurons)):
        for i in range(0, hiddens):
            x = tfkl.Dropout(0.3, seed=seed)(x)
            x = tfkl.Dense(
                neurons[i],
                activation='relu',
                kernel_initializer = tfk.initializers.GlorotUniform(seed))(x)
        outputs = tfkl.Dense(
            14,
            activation='softmax',
            kernel_initializer = tfk.initializers.GlorotUniform(seed))(x)

    # Connect input and output through the Model class
    tl_model = tfk.Model(inputs=inputs, outputs=outputs, name='model')
    return tl_model
    
    
    
def get_model_name(hiddens, neurons):
    neurons_str = ""
    for i in range(0, len(neurons)):
        neurons_str += str(neurons[i])
        if (len(neurons) > i + 1):
            neurons_str += '-'
    name_model = str(hiddens) + "h_" + neurons_str + "n"
    return name_model
    
    
    
def plot_history(history):
    '''
    this method plots the history of a model after training. The model must be trained with the metrics
    returned by the metrics() method
    '''
    # Plot the training
    patience = 10

    plt.figure(figsize=(15,5))
    plt.plot(history['loss'], label='Training', alpha=.3, color='#ff00ff')
    plt.plot(history['val_loss'], label='Validation', alpha=.8, color='#00ffff')
    plt.legend(loc='upper left')
    plt.title('Categorical Crossentropy')
    plt.grid(alpha=.3)

    plt.figure(figsize=(15,5))
    plt.plot(history['accuracy'], label='Training', alpha=.8, color='#ff00ff')
    plt.plot(history['val_accuracy'], label='Validation', alpha=.8, color='#00ffff')
    plt.legend(loc='upper left')
    plt.title('Accuracy')
    plt.grid(alpha=.3)

    plt.figure(figsize=(15,5))
    plt.plot(history['cat_acc'], label='Training', alpha=.8, color='#ff00ff')
    plt.plot(history['val_cat_acc'], label='Validation', alpha=.8, color='#00ffff')
    plt.legend(loc='upper left')
    plt.title('Categorical Accuracy')
    plt.grid(alpha=.3)

    plt.figure(figsize=(15,5))
    plt.plot(history['precision'], label='Training', alpha=.8, color='#ff00ff')
    plt.plot(history['val_precision'], label='Validation', alpha=.8, color='#00ffff')
    plt.legend(loc='upper left')
    plt.title('Precision')
    plt.grid(alpha=.3)

    plt.figure(figsize=(15,5))
    plt.plot(history['recall'], label='Training', alpha=.8, color='#ff00ff')
    plt.plot(history['val_recall'], label='Validation', alpha=.8, color='#00ffff')
    plt.legend(loc='upper left')
    plt.title('Recall')
    plt.grid(alpha=.3)

    plt.figure(figsize=(15,5))
    plt.plot(history['f1_m'], label='Training', alpha=.8, color='#ff00ff')
    plt.plot(history['val_f1_m'], label='Validation', alpha=.8, color='#00ffff')
    plt.legend(loc='upper left')
    plt.title('AUC')
    plt.grid(alpha=.3)

    plt.figure(figsize=(15,5))
    plt.plot(history['tp'], label='Training', alpha=.8, color='#ff00ff')
    plt.plot(history['val_tp'], label='Validation', alpha=.8, color='#00ffff')
    plt.legend(loc='upper left')
    plt.title('True Positive')
    plt.grid(alpha=.3)

    plt.figure(figsize=(15,5))
    plt.plot(history['fp'], label='Training', alpha=.8, color='#ff00ff')
    plt.plot(history['val_fp'], label='Validation', alpha=.8, color='#00ffff')
    plt.legend(loc='upper left')
    plt.title('False Positive')
    plt.grid(alpha=.3)

    plt.figure(figsize=(15,5))
    plt.plot(history['tn'], label='Training', alpha=.8, color='#ff00ff')
    plt.plot(history['val_tn'], label='Validation', alpha=.8, color='#00ffff')
    plt.legend(loc='upper left')
    plt.title('True Negative')
    plt.grid(alpha=.3)

    plt.figure(figsize=(15,5))
    plt.plot(history['fn'], label='Training', alpha=.8, color='#ff00ff')
    plt.plot(history['val_fn'], label='Validation', alpha=.8, color='#00ffff')
    plt.legend(loc='upper left')
    plt.title('False Negative')
    plt.grid(alpha=.3)

    plt.figure(figsize=(15,5))
    plt.plot(history['fn'], label='FN', alpha=.8, color='#ff00ff')
    plt.plot(history['fp'], label='FP', alpha=.8, color='#00ffff')
    plt.plot(history['tn'], label='TN', alpha=.8, color='#ff00ff')
    plt.plot(history['tp'], label='TP', alpha=.8, color='#00ffff')
    plt.legend(loc='upper left')
    plt.title('TP, TN, FP, FN training')
    plt.grid(alpha=.3)

    plt.figure(figsize=(15,5))
    plt.plot(history['fn'], label='FN', alpha=.8, color='#ff00ff')
    plt.plot(history['fp'], label='FP', alpha=.8, color='#00ffff')
    plt.plot(history['tn'], label='TN', alpha=.8, color='#ff00ff')
    plt.plot(history['tp'], label='TP', alpha=.8, color='#00ffff')
    plt.legend(loc='upper left')
    plt.title('TP, TN, FP, FN validation')
    plt.grid(alpha=.3)

    plt.figure(figsize=(15,5))
    plt.plot(history['prc'], label='Training', alpha=.8, color='#ff00ff')
    plt.plot(history['val_prc'], label='Validation', alpha=.8, color='#00ffff')
    plt.legend(loc='upper left')
    plt.title('PRC')
    plt.grid(alpha=.3)

    plt.figure(figsize=(15,5))
    plt.plot(history['auc'], label='Training', alpha=.8, color='#ff00ff')
    plt.plot(history['val_auc'], label='Validation', alpha=.8, color='#00ffff')
    plt.legend(loc='upper left')
    plt.title('AUC')
    plt.grid(alpha=.3)

    plt.figure(figsize=(15,5))
    plt.plot(history['loss'], label='Categorical Accuracy', alpha=.8, color='#ff00ff')
    plt.plot(history['accuracy'], label='Accuracy', alpha=.8, color='#00ffff')
    plt.plot(history['precision'], label='Precision', alpha=.8, color='#00ff00')
    plt.plot(history['recall'], label='Recall', alpha=.8, color='#ff0000')
    plt.legend(loc='upper left')
    plt.title('Training Metrics')
    plt.grid(alpha=.3)

    plt.show()
