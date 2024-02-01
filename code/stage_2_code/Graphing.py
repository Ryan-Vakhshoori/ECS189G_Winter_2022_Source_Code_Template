import matplotlib.pyplot as plt

class Graph():
    def traininglossgraph(self, epochs, train_loss):
        plt.plot(epochs, train_loss, label="Training Loss")
        plt.ylabel("Epochs")
        plt.xlabel("Training Loss")
        plt.show()