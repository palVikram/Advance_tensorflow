from keras.callbacks import ModelCheckpoint

# Define the file path pattern to save the model with the epoch number
filepath = 'model_checkpoint_{epoch:02d}.h5'

# Create the ModelCheckpoint callback
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_weights_only=False, save_best_only=False, save_freq='epoch')
