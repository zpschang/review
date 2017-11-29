class HyperParameter():
    def __init__(self):
        self.vocab_size = None
        self.feature_size = None
        self.embedding_size = 128
        self.rating_num = 6
        self.aspect_num = 6
        self.lstm_size = 200
        self.num_layer = 4
        self.num_type = 3
        self.num_rating = 6
        self.num_aspect = 6
        self.max_length = 15
        self.prefix_length = 3
        self.max_gradient_norm = 2
        self.batch_size = 20
        self.learning_rate = 0.001
        self.beam_width = 5

hyper = HyperParameter()