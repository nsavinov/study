import random
import math


class Register(object):
    # provides topological sorting by keeping track of the order
    # in which nodes are added
    def __init__(self):
        self.nodes = []

    def new(self, node):
        self.nodes += [node]

    def get_nodes(self):
        return self.nodes


class Op(object):
    register = Register()

    def __init__(self):
        Op.register.new(self)

    def forward(self):
        #assumes all inputs were already processed
        pass

    def backward(self):
        #assumes all outputs were already processed
        pass

    def accumulate(self):
        #call after backward
        pass

    def update(self, step):
        #call after processing mini-batch
        pass


class Value(Op):
    def forward(self):
        self.out_grad = .0
        self.out = self.val
        return self.out


class Param(Value):
    def __init__(self, init):
        super(Param, self).__init__()
        self.val = init

    def accumulate(self):
        self.grad += self.out_grad

    def update(self, step):
        self.val += step * self.grad


class Input(Value):
    def set(self, val):
        self.val = val


class Const(Value):
    def __init__(self, init):
        super(Const, self).__init__()
        self.val = init
        

class Plus(Op):
    def __init__(self, x, y):
        super(Plus, self).__init__()
        self.x = x
        self.y = y
    
    def forward(self):
        self.out_grad = .0
        self.out = self.x.out + self.y.out
        return self.out
        
    def backward(self):
        grad_x = self.out_grad
        grad_y = self.out_grad
        self.x.out_grad += grad_x
        self.y.out_grad += grad_y


class SplittingPlus(Op):
    def __init__(self, x, y):
        super(SplittingPlus, self).__init__()
        self.x = x
        self.y = y
        self.alpha = 0.01
        self.prob = 0.5
    
    def forward(self):
        self.out_grad = .0
        self.x_forward = self.x.out
        self.out = self.x.out + self.y.out
        return self.out
        
    def backward(self):
        grad_x = self.out_grad
        grad_y = self.out_grad
        if random.random() < self.prob:
            grad_x -= self.alpha * self.x_forward
            grad_y += self.alpha * self.x_forward
        self.x.out_grad += grad_x
        self.y.out_grad += grad_y


class Mult(Op):
    def __init__(self, x, y):
        super(Mult, self).__init__()
        self.x = x
        self.y = y
    
    def forward(self):
        self.out_grad = .0
        self.out = self.x.out * self.y.out
        return self.out
        
    def backward(self):
        grad_x = self.y.out * self.out_grad
        grad_y = self.x.out * self.out_grad
        self.x.out_grad += grad_x
        self.y.out_grad += grad_y


class ReLU(Op):
    def __init__(self, x):
        super(ReLU, self).__init__()
        self.x = x

    def forward(self):
        self.out_grad = .0
        self.out = self.x.out if self.x.out > .0 else .0

    def backward(self):
        grad_x = self.out_grad if self.x.out > .0 else .0
        self.x.out_grad = grad_x


class LeakyReLU(Op):
    def __init__(self, x):
        super(LeakyReLU, self).__init__()
        self.x = x
        self.a = 0.01

    def forward(self):
        self.out_grad = .0
        self.out = self.x.out if self.x.out > .0 else self.x.out * self.a

    def backward(self):
        grad_x = self.out_grad if self.x.out > .0 else self.out_grad * self.a
        self.x.out_grad = grad_x


class Sigmoid(Op):
    def __init__(self, x):
        super(Sigmoid, self).__init__()
        self.x = x

    def forward(self):
        self.out_grad = .0
        self.out = 1.0 / (1.0 + math.exp(-self.x.out))

    def backward(self):
        grad_x = self.out_grad * self.out * (1.0 - self.out)
        self.x.out_grad = grad_x


def get_random_init(positive=False):
    abs_init = 0.01
    if not positive:
        return (2. * random.random() - 1.) * abs_init 
    else:
        return abs_init


class ComputationalGraph:
    # only single output supported
    # all_nodes should be topologically sorted
    def __init__(self, inputs, output, all_nodes):
        self.inputs = inputs
        self.output = output
        self.all_nodes = all_nodes

    def print_all_nodes_out(self):
        print 'Node values:'
        for node in self.all_nodes:
            print type(node), node.out
        print '--------------------------'

    def print_all_nodes_out_grad(self):
        print 'Node gradients:'
        for node in self.all_nodes:
            print type(node), node.out_grad
        print '--------------------------'

    def set_inputs(self, inputs):
        assert len(inputs) == len(self.inputs)
        for index, val in enumerate(inputs):
            self.inputs[index].set(val)

    def forward(self):
        for node in self.all_nodes:
            node.forward()
        return self.output.out

    def backward(self):
        self.output.out_grad = self.out_grad
        for node in reversed(self.all_nodes):
            node.backward()

    def update_param_grad(self, step):
        for node in self.all_nodes:
            node.update(step)

    def accumulate_param_grad(self):
        for node in self.all_nodes:
            node.accumulate()

    def zero_param_grad(self):
        for node in self.all_nodes:
            node.grad = 0.0

    def set_out_grad(self, out_grad):
        self.out_grad = out_grad


def get_two_layer_mlp():
    # h1 = ReLU(ax1 * x1 + ax2 * x2 + ax) - first hidden
    # h2 = ReLU(bx1 * x1 + bx2 * x2 + bx) - second hidden
    # ah1 * h1 + ah2 * h2 + ah - output
    plus_op = Plus #SplittingPlus
    x1 = Input()
    x2 = Input()
    ########
    ax1 = Param(get_random_init())
    ax1x1 = Mult(ax1, x1)
    ax2 = Param(get_random_init())
    ax2x2 = Mult(ax2, x2)
    ax1x1_ax2x2 = Plus(ax1x1, ax2x2)
    ax = Param(get_random_init(True)) #True
    ax1x1_ax2x2_ax = plus_op(ax1x1_ax2x2, ax)
    h1 = ReLU(ax1x1_ax2x2_ax)
    ########
    bx1 = Param(get_random_init())
    bx1x1 = Mult(bx1, x1)
    bx2 = Param(get_random_init())
    bx2x2 = Mult(bx2, x2)
    bx1x1_bx2x2 = Plus(bx1x1, bx2x2)
    bx = Param(get_random_init(True)) #True
    bx1x1_bx2x2_bx = plus_op(bx1x1_bx2x2, bx)
    h2 = ReLU(bx1x1_bx2x2_bx)
    ########
    ah1 = Param(get_random_init())
    ah1h1 = Mult(ah1, h1)
    ah2 = Param(get_random_init())
    ah2h2 = Mult(ah2, h2)
    ah = Param(get_random_init())
    ah1h1_ah2h2 = Plus(ah1h1, ah2h2)
    output = plus_op(ah1h1_ah2h2, ah)
    ########
    return ComputationalGraph([x1, x2], output, Op.register.get_nodes())
        
        
def xor(lr):
    #TODO: why does is work so bad for XOR? any bugs?
    data = [[-1., -1.], [1., 1.], [-1., 1.], [1., -1.]]
    labels = [-1, -1, 1, 1]
    g = get_two_layer_mlp()
    print 'Start performance:'
    for index, _ in enumerate(data):
        g.set_inputs(data[index])
        prediction = g.forward()
        print data[index], labels[index], prediction
        g.print_all_nodes_out()
    print '------------------'
    g.zero_param_grad()
    for example in xrange(100000):
        index = random.randint(0, len(data) - 1)
        g.set_inputs(data[index])
        prediction = g.forward()
        g.set_out_grad(labels[index] - prediction)
        g.backward()
        g.accumulate_param_grad()
        if example % 10 == 0:
            g.update_param_grad(lr)
            g.zero_param_grad()
    print '------------------'
    print 'Final performance:'
    for index, _ in enumerate(data):
        g.set_inputs(data[index])
        prediction = g.forward()
        g.set_out_grad(labels[index] - prediction)
        g.backward()
        print data[index], labels[index], prediction
        g.print_all_nodes_out()
        g.print_all_nodes_out_grad()


xor(0.01)
