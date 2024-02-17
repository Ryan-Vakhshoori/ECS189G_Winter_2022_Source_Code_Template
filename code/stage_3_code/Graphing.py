import matplotlib.pyplot as plt

class Graph():
    def traininglossgraph(self, epochs, train_loss):
        plt.plot(range(1, epochs + 1), train_loss, label="Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Training Loss")
        plt.show()