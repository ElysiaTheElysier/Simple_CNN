import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Train a simple CNN on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    args = parser.parse_args()

    print(f"TensorFlow version: {tf.__version__}")

    # Load dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    print(f"Train shape: {x_train.shape}")
    print(f"Test shape: {x_test.shape}")

    # Normalize images
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Build model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("\nModel summary:")
    model.summary()

    # Train model
    print(f"\nTraining model for {args.epochs} epochs...")
    history = model.fit(x_train, y_train, epochs=args.epochs, batch_size=64, validation_data=(x_test, y_test))

    # Evaluate model
    test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
    print(f"\nFinal test accuracy: {test_acc:.4f}")

    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('assets', exist_ok=True)

    # Save model
    model_path = os.path.join('models', 'cnn_cifar10_model.keras')
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Save training accuracy chart
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.savefig(os.path.join('assets', 'training_accuracy.png'))
    plt.close()

    # Save training loss chart
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join('assets', 'training_loss.png'))
    plt.close()

    # Save sample predictions image
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_test[i])
        # Predict
        pred = model.predict(np.expand_dims(x_test[i], axis=0), verbose=0)
        pred_label = class_names[np.argmax(pred)]
        true_label = class_names[y_test[i][0]]
        color = 'blue' if pred_label == true_label else 'red'
        plt.xlabel(f"{pred_label} ({true_label})", color=color)
    plt.tight_layout()
    plt.savefig(os.path.join('assets', 'sample_predictions.png'))
    plt.close()
    
    print("Training charts and sample predictions saved to assets/")

if __name__ == "__main__":
    main()
