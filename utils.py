import numpy as np


def load_dataset():
    with np.load("mnist.npz") as f:
        x_train = f['x_train'].astype("float32") / 255
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))

        y_train = f['y_train']
        y_train = np.eye(10)[y_train]

    return x_train, y_train

def train(epochs, images, labels, learning_rate):
    weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (20, 784))
    weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (10, 20))

    bias_input_to_hidden = np.zeros((20, 1))
    bias_hidden_to_output = np.zeros((10, 1))

    e_loss = 0
    e_correct = 0

    for epoch in range(epochs):
        print(f"\nEpoch №{epoch+1}")
        for image, label in zip(images, labels):
            image = np.reshape(image, (-1, 1))
            label = np.reshape(label, (-1, 1))

            hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
            hidden = 1 / (1 + np.exp(-hidden_raw))

            output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
            output = 1 / (1 + np.exp(-output_raw))

            e_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=0)
            e_correct += int(np.argmax(output) == np.argmax(label))
        
            delta_output = output - label
            weights_hidden_to_output += -learning_rate * delta_output @ np.transpose(hidden)
            bias_hidden_to_output += -learning_rate * delta_output

            delta_hidden = np.transpose(weights_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
            weights_input_to_hidden += -learning_rate * delta_hidden @ np.transpose(image)
            bias_input_to_hidden += learning_rate * delta_hidden

        print(f"Loss: {round((e_loss[0]/images.shape[0]) * 100, 3)}%")
        print(f"Accuracy: {round((e_correct/images.shape[0]) * 100, 3)}%")
        e_loss = 0
        e_correct = 0

    return (weights_input_to_hidden, bias_input_to_hidden), (weights_hidden_to_output, bias_hidden_to_output)

import numpy as np

def save_model(filename, model):
    w1, b1 = model[0]
    w2, b2 = model[1]
    np.savez(filename, w1=w1, b1=b1, w2=w2, b2=b2)
    print(f"Model save in '{filename}.npz'")

def load_model(filename):
    data = np.load(f"{filename}.npz")
    w1 = data['w1']
    b1 = data['b1']
    w2 = data['w2']
    b2 = data['b2']
    print(f"Model loaded from: '{filename}.npz'")
    return (w1, b1), (w2, b2)

def predict(model, test_image):
    weights_input_to_hidden, bias_input_to_hidden = model[0]
    weights_hidden_to_output, bias_hidden_to_output = model[1]

    image = np.reshape(test_image, (-1, 1))

    hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
    hidden = 1 / (1 + np.exp(-hidden_raw))

    output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
    output = 1 / (1 + np.exp(-output_raw))

    return np.argmax(output)
