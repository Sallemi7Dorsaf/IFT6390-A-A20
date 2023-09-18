import numpy as np


class SVM:
    def __init__(self, eta, C, niter, batch_size,verbose):
        self.eta = eta
        self.C = C
        self.niter = niter
        self.batch_size = batch_size
        self.verbose = verbose

    def make_one_versus_all_labels(self, y, m):
        """
        y : numpy array of shape (n,)
        m : int (num_classes)
        returns : numpy array of shape (n, m)
        """
  
        a=(np.ones((len(y),m)))*-1
        for i,v in enumerate( y):
            a[i,v]=1
        
        return a

    def compute_loss(self, x, y):
      

        predictions=np.dot(x,self.w)

        indicateur=self.make_one_versus_all_labels(y, self.w.shape[1])
        a=np.multiply(indicateur,predictions)
        return a.mean()

        #return np.mean(np.maximum(0, 2 - np.multiply(indicateur,predictions)))+ (self.C/2)*np.sum(self.w ** 2)
  
 
      #return np.mean(np.maximum(0, 2 - y * np.dot(x,self.w)))+ (self.C/2)*np.sum(self.w ** 2) 
      #return np.mean(np.maximum(0, 2 - y * np.dot(x,self.w))) + (C/2)*np.sum(np.abs(self.w) ** 2)
        

    def compute_gradient(self, x, y):
        """
        x : numpy array of shape (minibatch size, num_features)
        y : numpy array of shape (minibatch size, num_classes)
        returns : numpy array of shape (num_features, num_classes)
        """
        dW = np.zeros(x.shape[0],y.shape[1]) # initialize the gradient as zero

        # compute the loss and the gradient
        num_classes = y.shape[1]
        num_train = x.shape[0]
        #loss = 0.0
        for i in xrange(num_train):
            scores = x[i].dot(y)
            correct_class_score = scores[y[i]]
            nb_sup_zero = 0
            for j in xrange(num_classes):
                if j == y[i]:
                    continue
                margin = scores[j] - correct_class_score + 1 # note delta = 1
                if margin > 0:
                    nb_sup_zero += 1
                    
                    dW[:, j] += X[i]
                dW[:, y[i]] -= nb_sup_zero*x[i]

  
        

        # average the gradient
        dW /= num_train


        return dW






    
    # Batcher function
    def minibatch(self, iterable1, iterable2, size=1):
        l = len(iterable1)
        n = size
        for ndx in range(0, l, n):
            index2 = min(ndx + n, l)
            yield iterable1[ndx: index2], iterable2[ndx: index2]

    def infer(self, x):
        """
        x : numpy array of shape (num_examples_to_infer, num_features)
        returns : numpy array of shape (num_examples_to_infer, num_classes)
        """
        predictions=np.dot(x,self.w)
        argmaxes=np.argmax(predictions,axis=1)
        return self.make_one_versus_all_labels(argmaxes,8)

    def compute_accuracy(self, y_inferred, y):
        """
        y_inferred : numpy array of shape (num_examples, num_classes)
        y : numpy array of shape (num_examples, num_classes)
        returns : float
        """
        
        return np.array([np.array_equal(y[i],y_inferred[i]) for i in range(len(y))]).mean()

    def fit(self, x_train, y_train, x_test, y_test):
        """
        x_train : numpy array of shape (number of training examples, num_features)
        y_train : numpy array of shape (number of training examples, num_classes)
        x_test : numpy array of shape (number of training examples, nujm_features)
        y_test : numpy array of shape (number of training examples, num_classes)
        returns : float, float, float, float
        """
        self.num_features = x_train.shape[1]
        self.m = y_train.max() + 1
        y_train = self.make_one_versus_all_labels(y_train, self.m)
        y_test = self.make_one_versus_all_labels(y_test, self.m)
        self.w = np.zeros([self.num_features, self.m])

        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []

        for iteration in range(self.niter):
            # Train one pass through the training set
            for x, y in self.minibatch(x_train, y_train, size=self.batch_size):
                grad = self.compute_gradient(x, y)
                self.w -= self.eta * grad

            # Measure loss and accuracy on training set
            train_loss = self.compute_loss(x_train, y_train)
            y_inferred = self.infer(x_train)
            train_accuracy = self.compute_accuracy(y_inferred, y_train)

            # Measure loss and accuracy on test set
            test_loss = self.compute_loss(x_test, y_test)
            y_inferred = self.infer(x_test)
            test_accuracy = self.compute_accuracy(y_inferred, y_test)

            if self.verbose:
                print(f"Iteration {iteration} | Train loss {train_loss:.04f} | Train acc {train_accuracy:.04f} |"
                      f" Test loss {test_loss:.04f} | Test acc {test_accuracy:.04f}")

            # Record losses, accs
            train_losses.append(train_loss)
            train_accs.append(train_accuracy)
            test_losses.append(test_loss)
            test_accs.append(test_accuracy)

        return train_losses, train_accs, test_losses, test_accs


# DO NOT MODIFY THIS FUNCTION
def load_data():
    # Load the data files
    print("Loading data...")
    x_train = np.load("train_features_cifar100_reduced.npz")["train_data"]
    x_test = np.load("test_features_cifar100_reduced.npz")["test_data"]
    y_train = np.load("train_labels_cifar100_reduced.npz")["train_label"]
    y_test = np.load("test_labels_cifar100_reduced.npz")["test_label"]

    # normalize the data
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # add implicit bias in the feature
    x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    x_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = load_data()

    print("Fitting the model...")
    svm = SVM(eta=0.0001, C=2, niter=200, batch_size=5000, verbose=False)
    train_losses, train_accs, test_losses, test_accs = svm.fit(x_train, y_train, x_test, y_test)

    # # to infer after training, do the following:
    # y_inferred = svm.infer(x_test)

    ## to compute the gradient or loss before training, do the following:
    # y_train_ova = svm.make_one_versus_all_labels(y_train, 8) # one-versus-all labels
    # svm.w = np.zeros([3073, 8])
    # grad = svm.compute_gradient(x_train, y_train_ova)
    # loss = svm.compute_loss(x_train, y_train_ova)
