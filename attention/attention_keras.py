from keras import backend as k
from keras.engine.topology import Layer

class Position_Embedding(Layer):

    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size, seq_len = k.shape(x)[0], k.shape(x)[1]
        postion_j = 1. / k.pow(10000., 2 * k.arange(self.size / 2, dtype='float32') / self.size)
        postion_j = k.expand_dims(postion_j, 0)
        postion_i = k.cumsum(k.ones_like(x[:, :, 0]), 1) - 1
        postion_i = k.expand_dims(postion_i, 2)
        postion_ij = k.dot(postion_i, postion_j)
        postion_ij = k.concatenate([k.cos(postion_ij), k.sin(postion_ij)], 2)

        if self.mode == 'sum':
            return postion_ij + x
        elif self.mode == 'concat':
            return k.concatenate([postion_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)

class Attention(Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = k.one_hot(seq_len[:, 0], k.shape(inputs)[1])
            mask = 1 - k.expand_dims(mask, 2)
            for _ in range(len(inputs.shape)-2):
                mask = k.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x

        # 对Q、K、V做线性变换
        Q_seq = k.dot(Q_seq, self.WQ)
        Q_seq = k.reshape(Q_seq, (-1, k.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = k.permute_dimensions(Q_seq, (0, 2, 1, 3))
        K_seq = k.dot(K_seq, self.WK)
        K_seq = k.reshape(K_seq, (-1, k.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = k.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = k.dot(V_seq, self.WV)
        V_seq = k.reshape(V_seq, (-1, k.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = k.permute_dimensions(V_seq, (0, 2, 1, 3))

        # 计算内积，然后mask，然后softmax
        A = k.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head ** 0.5
        A = k.permute_dimensions(A, (0, 3, 2, 1))
        A = self.Mask(A, V_len, 'add')
        A = k.permute_dimensions(A, (0, 3, 2, 1))
        A = k.softmax(A)

        # 输出并mask
        out_seq = k.batch_dot(A, V_seq, axes=[3, 2])
        out_seq = k.permute_dimensions(out_seq, (0, 2, 1, 3))
        out_seq = k.reshape(out_seq, (-1, k.shape(out_seq)[1], self.output_dim))
        out_seq = self.Mask(out_seq, Q_len, 'mul')

        return out_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)
