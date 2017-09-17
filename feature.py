from analyze import load_catagory

class feature_extractor():
    def __index__(self):
        self.catagory, self.word_to_catagory = load_catagory()

    def extract(self, batch):
        pass
