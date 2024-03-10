import matplotlib.pyplot as plt

class Graph():
    def traininglossgraph(self, epochs, train_loss):
        plt.plot(epochs, train_loss, label="Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Training Loss")
        plt.show()