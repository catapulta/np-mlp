import numpy as np


def load_mnist_data_file(path):
    files = ('train_data.npy',
            'train_labels.npy',
            'val_data.npy',
            'val_labels.npy',
            'test_data.npy',
            'test_labels.npy',)

    return [np.load(path+'{}'.format(file)) for file in files]


class Activation(object):
    """ Interface for activation functions (non-linearities).
    """

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):
    """ Identity function.
     """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):
    """ Sigmoid non-linearity.
    """

    def __init__(self):
        super(Sigmoid, self).__init__()
    
    # check if x should be negative
    def forward(self, x):
        self.state = 1.0 / (1.0 + np.exp(-x))
        return self.state
    
    def derivative(self):
        return self.state * (1 - self.state)


class Tanh(Activation):
    """ Implement the tanh non-linearity.
    """

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.state = np.tanh(x)
        return self.state

    def derivative(self):
        return (1 - self.state**2)


class ReLU(Activation):
    """ ReLU non-linearity.
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.state = np.maximum(0, x)
        return self.state

    def derivative(self):
        d = self.state.copy()
        d[self.state > 0] = 1
        return d


# CRITERION
class Criterion(object):
    """ Interface for loss functions.
    """

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):
    """ Softmax + CrossEntropy.
    """
    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None
        self.labels = None
        self.logits = None
        self.loss = None
    
    def softmax(self,x):
        max_num = np.max(x, axis=1, keepdims=True)
        exps = np.exp(x - max_num)
        sum_exps = np.sum(exps, axis=1, keepdims=True)
        return exps / sum_exps
    
    def forward(self, x, y):
        self.logits = self.softmax(x)
        if len(y.shape)>1:
            self.labels = y.argmax(axis=1)
        else:
            self.labels = y
        # take the y_i th value of each row
        losses = -np.log(self.logits[range(len(self.labels)), self.labels])
        # average by rows
        self.loss = np.sum(losses)
        return losses

    def derivative(self):
        dloss_dx = self.logits.copy()
        # substract 1 to true classes, 0 to false ones (y_i th value of each row )
        dloss_dx[range(len(self.labels)), self.labels] -= 1
        return dloss_dx


class BatchNorm(object):
    """ BatchNorm Layer.
    """
    def __init__(self, fan_in, alpha=0.9):
        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None
        self.fan_in = fan_in

        # The following attributes will be tested
        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((1, fan_in))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((1, fan_in))

        # inference parameters
        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        self.x = x
        n = len(x)
        x = x.ravel().reshape(n, -1)
        
        if eval:
            self.mean = self.running_mean
            self.var = self.running_var
        else:
            self.mean = np.mean(x, axis=0)
            self.var = np.var(x, axis=0)
            self.running_mean = self.alpha*self.running_mean + (1-self.alpha)*self.mean  
            self.running_var = self.alpha*self.running_var + (1-self.alpha)*self.var
            
        self.norm = (x - self.mean) / np.sqrt(self.var + self.eps)
        y = self.gamma * self.norm + self.beta
        return y

    def backward(self, delta):
        m = len(self.x)
        x = self.x.ravel().reshape(m, -1)
        
        self.dbeta = np.sum(delta, axis=0) 
        self.dgamma = np.sum(self.norm * delta, axis=0)
        
        # get dL/dx
        mean_div = x - self.mean
        adj_var = self.var + self.eps
        
        dnorm = delta * self.gamma
        dvar = -1/2 * np.sum(dnorm*mean_div, axis=0) * np.power(adj_var, (-3/2))
        dmean = np.sum(-dnorm * np.power(adj_var, (-1/2)), axis=0) - 2*dvar*np.mean(mean_div, axis=0)
        # dmean = np.sqrt(adj_var) - 1/2 * mean_div * adj_var**(-3/2)*-2/m*np.sum(mean_div,axis=0)
        # dmean = np.sum(dnorm * dmean, axis=0)
        dx = dnorm*np.power(adj_var, (-1/2)) + dvar*2/m*mean_div + dmean/m
        return dx

    def zero_grads(self):
        self.beta = np.zeros((1, self.fan_in))
        self.dbeta = np.zeros((1, self.fan_in))


def random_normal_weight_init(d0, d1):
    return np.random.randn(d0, d1)


def zeros_bias_init(d):
    return np.zeros(shape=(1,d))


class MLP(object):
    """ A simple multilayer perceptron.
    """

    def __init__(self, input_size, output_size, hiddens,
                 activations, weight_init_fn, bias_init_fn,
                 criterion, lr, momentum=0.0, num_bn_layers=0):
        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum

        layers_size = [self.input_size] + hiddens + [self.output_size]
        self.W = [weight_init_fn(layers_size[i], layers_size[i+1]) for i in range(len(layers_size)-1)]
        
        self.b = [bias_init_fn(hiddens[i]) for i in range(len(hiddens))] + [bias_init_fn(self.output_size)]
        
        self.dW = []
        self.db = []
        
        # if batch norm, add batch norm parameters
        if self.bn:
            layers_size = hiddens + [self.output_size]
            if len(hiddens) > self.num_bn_layers:
                self.bn_layers = [BatchNorm(layers_size[i]) for i in range(min(self.num_bn_layers, len(hiddens)))]
            else:
                raise Exception('Number of hidden layers is not enough for BatchNorm number.')
            
            if len(hiddens) + 1 == self.num_bn_layers:
                self.bn_layers = [BatchNorm(self.input_size)] + self.bn_layers

        self.x_list = None
        self.w_velocity = [np.zeros_like(w) for w in self.W]
        self.b_velocity = [np.zeros_like(b) for b in self.b]
        if self.bn:
            self.gamma_velocity = [np.zeros_like(bn.gamma) for bn in self.bn_layers]
            self.beta_velocity = [np.zeros_like(bn.beta) for bn in self.bn_layers]
        
        assert len(self.W) == len(self.b) and len(self.W) == len(self.activations)
        
    def forward(self, x):
        self.x_list = [x]
        for i, (w, b, activation) in enumerate(zip(self.W, self.b, self.activations)):
            layer_out = self.x_list[i] @ w + b
            if self.num_bn_layers > i:
                layer_out = self.bn_layers[i].forward(layer_out, eval=not self.train_mode)
            layer_out = activation.forward(layer_out)
            self.x_list.append(layer_out)
        
        return layer_out

    def zero_grads(self):
        self.dW = []
        self.db = []
        if self.num_bn_layers>0:
            for bn_layer in self.bn_layers:
                bn_layer.zero_grads()

    def step(self):
        for dW, W, db, b, vw, vb in zip(self.dW, self.W, self.db, self.b, self.w_velocity, self.b_velocity):
            vw *= self.momentum
            vw -= self.lr* dW
            W += vw
            
            vb *= self.momentum
            vb -= self.lr* db
            b += vb
            
        if self.bn:
            for bn_layer, vgamma, vbeta in zip(self.bn_layers, self.gamma_velocity, self.beta_velocity):
                vgamma *= self.momentum
                vgamma -= self.lr * bn_layer.dgamma
                bn_layer.gamma += vgamma

                vbeta *= self.momentum
                vbeta -= self.lr * bn_layer.dbeta
                bn_layer.beta += vbeta
                
    def backward(self, labels):
        self.criterion.forward(self.x_list[-1], labels)
        dy = self.criterion.derivative()
        for i in reversed(range(len(self.W))):
            dy = self.activations[i].derivative() * dy
            if self.num_bn_layers > i:
                dy = self.bn_layers[i].backward(dy)
            self.dW.append( self.x_list[i].T @ dy /len(dy) )
            self.db.append( np.sum(dy, axis=0) /len(dy) )
            dy = dy @ self.W[i].T
        self.dW = list(reversed(self.dW))
        self.db = list(reversed(self.db))
        
    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False


def minibatcher(x, y, batch_size, shuffle=True):
    m = x.shape[0]
    idx = np.arange(m)
    if shuffle:
        np.random.shuffle(idx)
        x = x[idx]
        if y is not None:
            y = y[idx]
    for i in range(0, len(x), batch_size):
        mini_x = x[i:i + batch_size]
        if y is not None:
            mini_y = y[i:i + batch_size]
            yield mini_x, mini_y
        else:
            yield mini_x, None
        

def get_training_stats(mlp, dset, nepochs, batch_size):
        ((train_data, train_labels),
        (val_data, val_labels),
        (test_data, test_labels)) = dset
        
        training_losses = np.zeros(nepochs)
        training_errors = np.zeros(nepochs)
        validation_losses = np.zeros(nepochs)
        validation_errors = np.zeros(nepochs)

        for epoch in range(nepochs):
            # training
            mlp.train_mode = True
            losses = 0
            errors = 0
            m = len(train_labels)
            idx = np.arange(m)
            np.random.shuffle(idx)
            x = train_data.copy()
            x = x[idx]
            y = train_labels.copy()
            y = y[idx]

            for i in range(0, m, batch_size):
                minix = x[i:i + batch_size]
                miniy = y[i:i + batch_size]
                mlp.zero_grads()
                pred = mlp.forward(minix)
                mlp.backward(miniy)
                mlp.step()
                losses += np.sum(mlp.criterion.loss)
                errors += np.sum(np.argmax(pred, axis=1)!=np.argmax(y, axis=1))

            training_losses[epoch] = losses / len(train_data)
            training_errors[epoch] = errors / len(train_data)
            print('Training Loss:', training_losses[epoch])
            print('Training Error:', training_errors[epoch])
            
            # validation
            mlp.train_mode = False
            losses = 0
            errors = 0
            mb_counts = 0
            for x, y in minibatcher(val_data, val_labels, batch_size):
                pred = mlp.forward(x)
                losses += np.sum(mlp.criterion.loss)
                errors += np.sum(np.argmax(pred, axis=1)!=np.argmax(y, axis=1))
                mb_counts += len(x)
            validation_losses[epoch] = losses / len(val_data)
            validation_errors[epoch] = errors / len(val_data)
            print('Validation Loss:', validation_errors[epoch])
            print('Validation Error:', validation_errors[epoch])
            print('---------------------------------------------------')
        
        mlp.train_mode = False
        preds = []
        for x, y in minibatcher(test_data, None, batch_size):
            pred = mlp.forward(x)
            pred = np.argmax(pred, axis=1)
            pred = make_one_hot(pred)
            preds.append(pred)

        # one hot encode preds
        preds = np.concatenate(preds)
        confusion_matrix = ( preds.T @ test_labels ).astype(int)
        return training_losses, training_errors, validation_losses, validation_errors, confusion_matrix


def make_one_hot(labels_idx):
    labels = np.zeros((labels_idx.shape[0], 10))
    labels[np.arange(labels_idx.shape[0]), labels_idx] = 1
    return labels


def process_dset_partition(dset_partition):
    data, labels_idx = dset_partition
    mu, std = data.mean(), data.std()
    # mu, std = 0, 1
    return (data - mu) / std, make_one_hot(labels_idx)


if __name__ == '__main__':
    train_data_path = "../data/train_data.npy"
    train_labels_path = "../data/train_labels.npy"

    val_data_path = "../data/val_data.npy"
    val_labels_path = "../data/val_labels.npy"

    test_data_path = "../data/test_data.npy"
    test_labels_path = "../data/test_labels.npy"
    
    mlp = MLP(784, 10, [64, 64, 32], [Sigmoid(), Sigmoid(), Sigmoid(), Identity()], random_normal_weight_init,
        zeros_bias_init, SoftmaxCrossEntropy(), 0.008, momentum=0.9, num_bn_layers=0)
    
    mlp = MLP(784, 10, [64, 32], [Sigmoid(), Sigmoid(), Identity()],
              random_normal_weight_init, zeros_bias_init, SoftmaxCrossEntropy(), 0.08,
              momentum=0.0, num_bn_layers=0)

    train_data, train_labels_idx = (np.load(train_data_path), np.load(train_labels_path))
    train_data, train_labels_idx = train_data[:10002], train_labels_idx[:10002]
    train_labels = np.zeros((train_labels_idx.shape[0], 10))
    train_labels[np.arange(train_labels_idx.shape[0]), train_labels_idx] = 1

    test_data, test_labels_idx = (np.load(test_data_path), np.load(test_labels_path))
    test_labels = np.zeros((test_labels_idx.shape[0], 10))
    test_labels[np.arange(test_labels_idx.shape[0]), test_labels_idx] = 1

    preds = get_training_stats(mlp, (
        (train_data, train_labels),
        (train_data, train_labels),
        (test_data, test_labels)), 4, 100)
    print(preds)
