import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from PIL import Image
import numpy as np

# Load pre-trained ResNet model without the top (classification) layer
base_model = tf.keras.applications.ResNet50(weights='imagenet')
base_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

# Load pre-trained image captioning model (LSTM-based)
caption_model = load_model('path_to_caption_model.h5')  # You should replace this with the actual path

# Load Tokenizer and vocabulary
tokenizer = Tokenizer()
with open('path_to_vocab.txt', 'r') as file:
    vocab = file.read().splitlines()
tokenizer.word_index = {word: i for i, word in enumerate(vocab)}
tokenizer.index_word = {i: word for i, word in enumerate(vocab)}

# Function to preprocess image for ResNet model
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to generate captions
def generate_caption(img_path):
    # Preprocess image
    img_array = preprocess_image(img_path)
    
    # Get image features using the pre-trained ResNet model
    img_features = base_model.predict(img_array)
    
    # Initialize caption input sequence
    caption_input_seq = np.zeros((1, 1))
    
    # Initialize caption
    caption = []
    
    # Set the start token
    start_token = tokenizer.word_index['<start>']
    
    # Set maximum caption length
    max_caption_length = 20
    
    for _ in range(max_caption_length):
        # Predict next word using the pre-trained caption model
        caption_probs = caption_model.predict([img_features, caption_input_seq])
        
        # Get the index of the predicted word
        predicted_word_index = np.argmax(caption_probs)
        
        # Map the index to the word
        predicted_word = tokenizer.index_word[predicted_word_index]
        
        # Break if the end token is encountered
        if predicted_word == '<end>':
            break
        
        # Update the caption
        caption.append(predicted_word)
        
        # Update the input sequence for the next iteration
        caption_input_seq[0, 0] = predicted_word_index
    
    return ' '.join(caption)

# Example usage
img_path = 'path_to_image.jpg'  # You should replace this with the actual path
caption = generate_caption(img_path)
print(f"Image Caption: {caption}")

from keras.applications import ResNet50
from keras.models import Model
from keras.layers import LSTM, Embedding, Dense, Input
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# Load and configure the ResNet50 model for feature extraction
resnet = ResNet50(weights='imagenet', include_top=False)
image_input = Input(shape=(224, 224, 3))
features = resnet(image_input)

# Define the RNN/LSTM part for caption generation
caption_input = Input(shape=(max_caption_length,))
caption_embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim)(caption_input)
lstm_output = LSTM(units)(caption_embedding)
caption_output = Dense(vocabulary_size, activation='softmax')(lstm_output)

# Combine into a single model
model = Model(inputs=[image_input, caption_input], outputs=caption_output)

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit([image_data, padded_sequences], word_sequences, epochs=20)

# Predict captions for new images
