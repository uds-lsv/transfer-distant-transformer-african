import os
import torch
import logging

import numpy as np
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


class Sentence:

    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels


class SentenceRepresentation:
    """
    All the information for a sentence needed by BERT, i.e.
    token_id, label_id + token_type, padding, masking
    """

    def __init__(self, token_ids, token_type_ids, attention_masks, label_ids):
        self.token_ids = token_ids
        self.token_type_ids = token_type_ids
        self.attention_masks = attention_masks
        self.label_ids = label_ids


def read_sentences_from_file(data_dir, filename, delimiter="\t"):
    filepath = os.path.join(data_dir, filename)
    sentences = []
    current_tokens = []
    current_labels = []
    with open(filepath, "r") as input_file:
        for line in input_file:
            line = line.strip()

            # end of current sentence, start of new sentence
            if line.startswith("-DOCSTART-") or len(line) == 0:
                if len(current_tokens) > 0:
                    assert len(current_tokens) == len(current_labels)
                    sentences.append(Sentence(current_tokens, current_labels))
                    current_tokens = []
                    current_labels = []
            else:
                line = line.split(delimiter)
                assert len(line) > 1, f"Line {line} could not be split into token and label. Maybe the delimiter is" \
                                      f"incorrect? Or the line does not contain a token and a label?"
                assert len(line[0]) > 0, f"Line '{line}' has no token"
                assert len(line[-1]) > 0, f"Line '{line}' has no label"
                if line[0] == "\u200b" or line[0] == "\ufeff":
                    logger.warning(f"Dataset contains a token that is a zero-width-space "
                                   f"({line[0].encode('raw_unicode_escape')}) but has a label. Ignoring this token.")
                    continue
                current_tokens.append(line[0]) # first element is always token
                current_labels.append(line[-1]) # we assume that last element is label (usually true)

        if len(current_tokens) > 0:
            assert len(current_tokens) == len(current_labels)
            sentences.append(Sentence(current_tokens, current_labels))

    return sentences


def split_too_long_sentences(sentences, tokenizer, max_seq_length):
    # the tokenizer adds additional tokens (e.g. [CLS], see convert_ function below)
    # correct max_seq_length for that
    max_seq_length = max_seq_length - tokenizer.num_added_tokens()

    new_sentences = []
    counter = 0
    for old_sentence in sentences:
        sentence_start_i = 0
        sentence_length = 0

        for i, token in enumerate(old_sentence.tokens):
            token_length = len(tokenizer.tokenize(token))
            if sentence_length + token_length > max_seq_length:
                new_tokens = old_sentence.tokens[sentence_start_i:i] # without current token as adding would be too long
                new_labels = old_sentence.labels[sentence_start_i:i]
                assert len(new_tokens) == len(new_labels)
                new_sentences.append(Sentence(new_tokens, new_labels))
                sentence_start_i = i
                sentence_length = 0
                counter += 1
            sentence_length += token_length

        if sentence_start_i == 0:  # original sentence long enough, do not need splitting
            new_sentences.append(old_sentence)
        else:  # was splitted, add the rest as a new sentence
            new_tokens = old_sentence.tokens[sentence_start_i:]  # till end of old sentence
            new_labels = old_sentence.labels[sentence_start_i:]
            new_sentences.append(Sentence(new_tokens, new_labels))

    if counter > 0:
        logger.info(f"Split sentences {counter} times as they were longer than the max_seq_length {max_seq_length}")
    return new_sentences


def create_label_map(label_list):
    return {label: i for i, label in enumerate(label_list)}

def convert_sentences_to_features_and_labels(sentences, label_list, max_seq_length, tokenizer,
                                             cls_token_at_end=False,
                                             cls_token="[CLS]",
                                             cls_token_segment_id=0,
                                             sep_token="[SEP]",
                                             sep_token_extra=False,
                                             pad_token=0,
                                             pad_token_segment_id=0,
                                             pad_token_label_id=-100,
                                             sequence_a_segment_id=0):
    label_map = create_label_map(label_list)

    sentence_representations = []
    for sentence_idx, sentence in enumerate(sentences):

        tokens = []
        label_ids = []
        for word, label in zip(sentence.tokens, sentence.labels):
            word_tokens = tokenizer.tokenize(word) # BERT has sub-word tokens, so we need to tokenize each word
            assert len(word_tokens) > 0, f"The tokenizer returned an empty word for the word '{word}' " \
                                         f"({word.encode('raw_unicode_escape')})."
            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = tokenizer.num_added_tokens()
        if len(tokens) > max_seq_length - special_tokens_count:
            raise Exception(f"Sentence {sentence.tokens} too long.")
            #tokens = tokens[: (max_seq_length - special_tokens_count)]
            #label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        # while the code below was used in the Huggingface NER code for the Roberta model,
        # this creates sequences that end with two </s> </s> which is not what is generated
        # when calling tokenizer.encode_plus
        if sep_token_extra:
            assert tokenizer.num_added_tokens() == 2
            # roberta uses an extra separator b/w pairs of sentences
            #tokens += [sep_token]
            #label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids += [pad_token] * padding_length
        input_mask += [0] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length, f"{len(input_ids)} {max_seq_length} {input_ids} {sep_token} {tokenizer.cls_token}"
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length, f"{len(label_ids)} {max_seq_length} {label_ids} {tokens} {input_ids}"

        if sentence_idx < 3:
            logger.info("Tokenization example")
            logger.info(f"  text: {sentence.tokens}")
            logger.info(f"  tokens (by input): {[tokenizer.tokenize(word) for word in sentence.tokens]}")
            logger.info(f"  tokens: {tokens}")
            logger.info(f"  token_ids: {input_ids}")
            logger.info(f"  token_type_ids: {segment_ids}")
            logger.info(f"  attention mask: {input_mask}")
            logger.info(f"  labels: {sentence.labels}")
            logger.info(f"  labels: {label_ids}")
        sentence_representations.append(
            SentenceRepresentation(token_ids=input_ids, token_type_ids=segment_ids,
                                   attention_masks=input_mask, label_ids=label_ids)
        )
    return sentence_representations


def load_dataset(filename, tokenizer, exp_config, subset_indices=None, return_sentences=False):
    """
    :param filename:
    :param tokenizer:
    :param exp_config:
    :param subset_indices: the indices of sentences (not tokens!)
    :return:
    """
    print("Data dir")
    print(exp_config["data_dir"])
    print(filename)
    sentences = read_sentences_from_file(exp_config["data_dir"], filename)

    if subset_indices is not None:
        sentences = np.array(sentences)[subset_indices]
        logging.info(f"Loading subset of dataset with indices {subset_indices} from {filename}.")
    else:
        logging.info(f"Loading full dataset from {filename}.")
    logging.info(f"Loaded {len(sentences)} sentences.")

    sentences = split_too_long_sentences(sentences, tokenizer, exp_config["max_seq_length"])

    sep_token_extra = "roberta" in exp_config["model_name"]
    sent_reprs = convert_sentences_to_features_and_labels(sentences, exp_config["labels"],
                                                          exp_config["max_seq_length"], tokenizer,
                                                          cls_token=tokenizer.cls_token,
                                                          sep_token=tokenizer.sep_token,
                                                          sep_token_extra=sep_token_extra,
                                                          pad_token=tokenizer.pad_token_id,
                                                          pad_token_segment_id=tokenizer.pad_token_type_id
                                                          )

    # Convert to Tensors and build dataset
    all_token_ids = torch.tensor([sent_repr.token_ids for sent_repr in sent_reprs], dtype=torch.long)
    all_token_type_ids = torch.tensor([sent_repr.token_type_ids for sent_repr in sent_reprs], dtype=torch.long)
    all_attention_masks = torch.tensor([sent_repr.attention_masks for sent_repr in sent_reprs], dtype=torch.long)
    all_label_ids = torch.tensor([sent_repr.label_ids for sent_repr in sent_reprs], dtype=torch.long)

    dataset = TensorDataset(all_token_ids, all_token_type_ids, all_attention_masks, all_label_ids)

    if return_sentences:
        return dataset, sentences

    return dataset


def batch_to_inputs(batch, exp_config):
    """
    Convert the output of the DataLoader to the inputs expected by the model.
    :return:
    """
    batch = tuple(t.to(exp_config["device"]) for t in batch)
    inputs = {"input_ids": batch[0], "attention_mask": batch[2]}
    # XLM and RoBERTa don't use segment_ids/token_types
    if not "roberta" in exp_config["model_name"] and not "xlm" in exp_config["model_name"]\
            and not "distilbert" in exp_config["model_name"]:
        inputs["token_type_ids"] = batch[1]
    labels = batch[3]
    return inputs, labels


if __name__ == "__main__":
    """
    example_sentences = read_sentences_from_file("../../data/conll_en/", "train.txt", delimiter="\t")
    print(example_sentences[0].tokens)
    print(example_sentences[0].labels)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    convert_sentences_to_features_and_labels(example_sentences, ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG",
                                                         "B-MISC", "I-MISC", "O"], 100, tokenizer)
    """
    """
    a_sent = Sentence(["A", "carolingual", "mountain"], ["A", "B", "C"])
    b_sent = Sentence(["A", "black", "car"], ["D", "E", "F"])
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    new_sents = split_too_long_sentences([a_sent, b_sent], tokenizer, 4)

    for new_sent in new_sents:
        print(new_sent.tokens)
        print(new_sent.labels)
    """