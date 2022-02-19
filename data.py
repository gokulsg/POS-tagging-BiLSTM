# import nltk
import conllu
import torch
from torch.utils.data import Dataset, DataLoader
import torchtext as tt
from itertools import chain
import tqdm

class TaggingDataset(Dataset):
    """
    A Pytorch dataset representing a tagged corpus in CoNLL format.
    Each item is a dictionary {'sentence': ..., 'tags': ...} representing one tagged sentence.
    The values under 'sentence' and 'tags' are int tensors of shape (sequence_length,).
    """
    def __init__(self, sentences, tag_sequences):
        self.sentences = sentences
        self.tag_sequences = tag_sequences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sample = {"sentence": self.sentences[idx], "tags": self.tag_sequences[idx]}
        return sample

def tagging_collate_fn(batch):
    tensors = []
    for instance in batch:
        sent_t = instance["sentence"]
        pos_t = instance["tags"]
        tensors.append(torch.stack([sent_t, pos_t]))

    return torch.stack(tensors)

def preprocess_token(word):
    if word == ".":
        return "</s>"
    else:
        return word.lower()

# create vocabularies
def unk_init(x):
    return torch.randn_like(x)

def load_conllu_corpus(filename):
    sentences = list()
    tag_sequences = list()

    data_file = open(filename, "r", encoding="utf-8")
    for tokenlist in conllu.parse_incr(data_file):
        sentence = [preprocess_token(tok["form"]) for tok in tokenlist]
        tags = [tok["upos"] for tok in tokenlist]
        sentences.append(sentence)
        tag_sequences.append(tags)

    return sentences, tag_sequences

def make_dataloader(sentences, tag_sequences, sents_vocab, pos_vocab):
    train_sent_ts = [torch.tensor(sents_vocab.lookup_indices(sent)) for sent in sentences]
    train_tag_ts = [torch.tensor(pos_vocab.lookup_indices(seq)) for seq in tag_sequences]

    train_dataset = TaggingDataset(train_sent_ts, train_tag_ts)
    train_dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=tagging_collate_fn)

    return train_dataloader

def load(training_corpus, development_corpus, test_corpus):
    """
    Loads a tagged corpus in CoNLL format and returns a tuple with the following entries:
    - DataLoader for the training set
    - DataLoader for the development set
    - DataLoader for the test set
    - Vocabulary for the natural-language side (as a Torchtext Vocab object)
    - Vocabulary for the tag side (dito)
    - Pretrained embeddings, as a tensor of shape (vocabulary size, embedding dimension)

    :param training_corpus: filename of the training corpus
    :param development_corpus: filename of the development corpus
    :param test_corpus: filename of the test corpus
    :return:
    """
    # read CoNLL file "de_gsd-ud-train.conllu"
    train_sentences, train_tag_sequences = load_conllu_corpus(training_corpus)
    dev_sentences, dev_tag_sequences = load_conllu_corpus(development_corpus)
    test_sentences, test_tag_sequences = load_conllu_corpus(test_corpus)

    # Build vocabulary from pretrained word embeddings.
    # Theoretically, FastText should predict embeddings for unknown words using subword embeddings;
    # but this does not seem to work when using it through Torchtext. Maybe I should fix this sometime.
    fasttext = tt.vocab.FastText(language='de', unk_init=unk_init)
    sents_vocab = tt.vocab.build_vocab_from_iterator(chain(train_sentences, dev_sentences, test_sentences), specials=["<unk>", "<pad>"])
    sents_vocab.set_default_index(0)
    pretrained_embeddings = fasttext.get_vecs_by_tokens(sents_vocab.get_itos())
    pos_vocab = tt.vocab.build_vocab_from_iterator(chain(train_tag_sequences, dev_tag_sequences, test_tag_sequences))

    # Map corpora to datasets
    train_dataloader = make_dataloader(train_sentences, train_tag_sequences, sents_vocab, pos_vocab)
    dev_dataloader = make_dataloader(dev_sentences, dev_tag_sequences, sents_vocab, pos_vocab)
    test_dataloader = make_dataloader(test_sentences, test_tag_sequences, sents_vocab, pos_vocab)

    return train_dataloader, dev_dataloader, test_dataloader, sents_vocab, pos_vocab, pretrained_embeddings

