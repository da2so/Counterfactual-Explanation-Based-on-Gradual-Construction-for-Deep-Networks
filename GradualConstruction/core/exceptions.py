class NoSuchMethodFileError(Exception):
    def __init__(self, message):
        self.message = message

class NoSuchMethodError(Exception):
    def __init__(self, message):
        self.message = message