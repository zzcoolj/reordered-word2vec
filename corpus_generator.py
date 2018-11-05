import os


class WikiSentences(object):
    """
    WikiSentences class modified based on the code from https://rare-technologies.com/word2vec-tutorial/
    """

    def __init__(self, dirname, units=None):
        """
        :param dirname:
        :param units: 0: use all data in dir
        :param random_selection:
        """
        self.dirname = dirname
        # wiki data has folders like 'AA', 'AB', ..., 'EJ', one unit stands for one of these folders.
        self.sub_folder_names = [sub_folder_name for sub_folder_name in os.listdir(self.dirname)
                                 if not sub_folder_name.startswith('.')]
        if units:
            # if random_selection:
            #     self.sub_folder_names = random.sample(self.sub_folder_names, units)
            # else:
            #     # get last units number of elements; the original list is like ['AB', 'AA']
            #     self.sub_folder_names = self.sub_folder_names[-units:]
            self.sub_folder_names = units
            print('ATTENTION: Only part of the corpus is used.', self.sub_folder_names)

    def __iter__(self):
        for sub_folder_name in self.sub_folder_names:
            sub_folder_path = os.path.join(self.dirname, sub_folder_name)
            for fname in os.listdir(sub_folder_path):
                if not fname.startswith('.'):
                    for line in open(os.path.join(sub_folder_path, fname), 'r', encoding='utf-8'):
                        yield line.strip()


with open('input/test.txt', 'w') as f:
    count = 0
    for sent in WikiSentences(dirname='/vol/corpusiles/open/Wikipedia-Dumps/en/20170420/prep/', units=['AA']):
        f.write(sent+'\n')
        count += 1
    print(count)
