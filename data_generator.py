import numpy as np
from utilities import load_data, calculate_scalar, scale


# class DataGenerator(object):
#     def __init__(self, batch_size, type, te_max_iter=None):
#         assert type in ['train', 'test']
#         self._batch_size_ = batch_size
#         self._type_ = type
#         self._te_max_iter_ = te_max_iter
#         self.rs = np.random.RandomState(3492)
#         
#     def generate(self, xs, ys):
#         x = xs[0]
#         y = ys[0]
#         batch_size = self._batch_size_
#         n_samples = len(x)
#         
#         index = np.arange(n_samples)
#         self.rs.shuffle(index)
#         
#         iter = 0
#         epoch = 0
#         pointer = 0
#         while True:
#             if (self._type_ == 'test') and (self._te_max_iter_ is not None):
#                 if iter == self._te_max_iter_:
#                     break
#             iter += 1
#             if pointer >= n_samples:
#                 epoch += 1
#                 if (self._type_) == 'test' and (epoch == 1):
#                     break
#                 pointer = 0
#                 self.rs.shuffle(index)                
#  
#             batch_idx = index[pointer : min(pointer + batch_size, n_samples)]
#             pointer += batch_size
#             yield x[batch_idx], y[batch_idx]
            
            
            
class DataGenerator(object):

    def __init__(self, batch_size, seed=1234):
        """
        Inputs:
          hdf5_path: str
          batch_size: int
          dev_train_csv: str | None, if None then use all data for training
          dev_validate_csv: str | None, if None then use all data for training
          seed: int, random seed
        """

        self.batch_size = batch_size

        self.random_state = np.random.RandomState(seed)
        self.validate_random_state = np.random.RandomState(0)

        # Load data
        
        (self.train_x, self.train_y, self.validate_x, self.validate_y, _, _) = load_data()
        
        self.train_audio_names = np.arange(len(self.train_x))
        self.validate_audio_names = np.arange(len(self.validate_x))
        
        # Calculate scalar
        (self.mean, self.std) = calculate_scalar(self.train_x)

    def generate_train(self):
        """Generate mini-batch data for training. 
        
        Returns:
          batch_x: (batch_size, feature_num)
          batch_y: (batch_size,)
        """

        batch_size = self.batch_size
        audios_num = len(self.train_x)
        audio_indexes = np.arange(audios_num)

        self.random_state.shuffle(audio_indexes)

        iteration = 0
        pointer = 0

        while True:

            # Reset pointer
            if pointer >= audios_num:
                pointer = 0
                self.random_state.shuffle(audio_indexes)

            # Get batch indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + batch_size]
            pointer += batch_size

            iteration += 1

            batch_x = self.train_x[batch_audio_indexes]
            batch_y = self.train_y[batch_audio_indexes]

            # Transform data
            batch_x = self.transform(batch_x)

            yield batch_x, batch_y

    def generate_validate(self, data_type, shuffle, max_iteration=None):
        """Generate mini-batch data for evaluation. 
        
        Returns:
          batch_x: (batch_size, seq_len, freq_bins)
          batch_y: (batch_size,)
          batch_audio_names: (batch_size,)
        """

        if data_type == 'train':
            x = self.train_x
            y = self.train_y
            audio_names = self.train_audio_names
            
        elif data_type == 'validate':
            x = self.validate_x
            y = self.validate_y
            audio_names = self.validate_audio_names

        batch_size = self.batch_size
        audios_num = len(x)
        audio_indexes = np.arange(audios_num)
            
        if shuffle:
            self.validate_random_state.shuffle(audio_indexes)
            
        iteration = 0
        pointer = 0

        while True:

            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= audios_num:
                break

            # Get batch indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + batch_size]
                
            pointer += batch_size

            iteration += 1

            batch_x = x[batch_audio_indexes]
            batch_y = y[batch_audio_indexes]
            batch_audio_names = audio_names[batch_audio_indexes]

            # Transform data
            batch_x = self.transform(batch_x)

            yield batch_x, batch_y, batch_audio_names

    def transform(self, x):
        """Transform data. 
        
        Args:
          x: (batch_x, seq_len, freq_bins) | (seq_len, freq_bins)
          
        Returns:
          Transformed data. 
        """

        return scale(x, self.mean, self.std)
        
        
class TestDataGenerator(DataGenerator):
    
    def __init__(self, batch_size):
        """Data generator for test data. 
        """
        
        super(TestDataGenerator, self).__init__(batch_size=batch_size)
        
        # Load test data
        (_, _, _, _, self.test_x, self.test_y) = load_data()
        
        self.test_audio_names = np.arange(len(self.test_x))
        
    def generate_test(self):
        
        audios_num = len(self.test_x)
        audio_indexes = np.arange(audios_num)
        batch_size = self.batch_size
        
        pointer = 0
        
        while True:

            # Reset pointer
            if pointer >= audios_num:
                break

            # Get batch indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + batch_size]
                
            pointer += batch_size

            batch_x = self.test_x[batch_audio_indexes]
            batch_audio_names = self.test_audio_names[batch_audio_indexes]

            # Transform data
            batch_x = self.transform(batch_x)

            yield batch_x, batch_audio_names