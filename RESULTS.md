## Hyperparameter tuning för CNN
**Resultaten finns i 5_2_cnn_filter_size_dropout**

```
global_model_type = 'tune_cnn'
global_folder_name = 'weyo4'
file_suffix = 'weyo1'
global_batch_size = 32
global_epochs = 50
global_learning_rate = 0.0001
own_file_path = os.getcwd() 
global_hyperparameter_folder_name = 'real_cnn_filter_size_dropout1'
optimized_model = False # Om satt till True körs den hyperparameter tune:ade modellen. Annars base_model
``` 

Vi tune:ar här antalet filter för de två cnn-lagren i mitten som prövar tre värden (32, 64, 128). Vi tune:ar också
antalet filter för det sista lagret som prövar värdena (32, 64, 128, 256). Vi tune:ar också dropout för samtliga lager(de är alla kopplade till samma variabel). Dropouten går mellan 0.1 och 0.5 med 0.1 som step.

```
def hyperparameter_cnn_model(hp):
    """
    Function that builds a CNN model with optimized hyperparameters including filter size and dropout rate
    """

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), 
                    activation='relu', padding='same', input_shape=(126, 13, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3), 
                    activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))

    model.add(Conv2D(filters = hp.Choice('conv_2_filters_1', values = [32, 64, 128]), kernel_size=(3, 3),
                    activation='relu', padding='same'))
    model.add(Conv2D(filters = hp.Choice('conv_2_filters_1', values = [32, 64, 128]), kernel_size=(3, 3),
                    activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Conv2D(filters = hp.Choice('conv_2_filters_2', values = [32, 64, 128, 256]), kernel_size=(3, 3),
                    activation='relu', padding='same'))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=global_learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
```

## Hyperparameter tuning för LSTM
**Resultaten finns i 5_2_lstm_filter_size_dropout**
Notera skillnaden i learning rate här mellan CNN och LSTM.

```
global_model_type = 'tune_lstm'
global_folder_name = '5_2_lstm_filter_size_dropout'
file_suffix = 'weyo1'
global_batch_size = 32
global_epochs = 50
global_learning_rate = 0.001
own_file_path = os.getcwd() 
global_hyperparameter_folder_name = '5_2_lstm_filter_size_dropout' #lstm_filter_size_dropout1' #'cnn_filter_size_dropout'
optimized_model = False # Om satt till True körs den hyperparameter tune:ade modellen. Annars base_model

```
```
def hyperparameter_lstm_model(hp):
    """
    Function that builds a LSTM model with optimized hyperparameters including filter size and dropout rate
    """
    
    model = Sequential()
    #model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True), input_shape=(126, 128)))
    #model.add(CuDNNLSTM(4, input_shape=(126, 128), return_sequences=True))
    #model.add(Dropout(0.5))
    model.add(LSTM(32, input_shape=(126, 13), return_sequences=True))
    model.add(Dropout(hp.Float('dropout_lstm', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(LSTM(units = hp.Choice('lstm_layer_units_1', values = [32, 64, 128]), return_sequences=True))
    model.add(Dropout(hp.Float('dropout_lstm', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(LSTM(units = hp.Choice('lstm_layer_units_1', values = [32, 64, 128]), return_sequences=True))
    model.add(Dropout(hp.Float('dropout_lstm', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(LSTM(units = hp.Choice('lstm_layer_units_2', values = [32, 64, 128]), return_sequences=True))
    model.add(Dropout(hp.Float('dropout_lstm', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(LSTM(units = hp.Choice('lstm_layer_units_2', values = [32, 64, 128]), return_sequences=True))
    model.add(Dropout(hp.Float('dropout_lstm', min_value=0.1, max_value=0.5, step=0.1)))
    #model.add(LSTM(16, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=global_learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
```

## Resultat för den konventionella CNN-modellen (ingen hyperparameter tuning)
```
global_model_type = 'cnn'
global_folder_name = 'cnn_base_model'
file_suffix = 'weyo1'
global_batch_size = 32
global_epochs = 50
global_learning_rate = 0.0001
own_file_path = os.getcwd() 
global_hyperparameter_folder_name = '5_2_lstm_filter_size_dropout' #lstm_filter_size_dropout1' #'cnn_filter_size_dropout'
optimized_model = False # Om satt till True körs den hyperparameter tune:ade modellen. Annars base_model

```
## Resultat för den optimerade CNN-modellen
```
global_model_type = 'cnn'
global_folder_name = 'cnn_optimized_model'
file_suffix = 'weyo1'
global_batch_size = 32
global_epochs = 50
global_learning_rate = 0.0001
own_file_path = os.getcwd() 
global_hyperparameter_folder_name = '5_2_lstm_filter_size_dropout' #lstm_filter_size_dropout1' #'cnn_filter_size_dropout'
optimized_model = True # Om satt till True körs den hyperparameter tune:ade modellen. Annars base_model
```