import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel


class BERT:

    def __init__(self):
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        # Load pre-trained model (weights)
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased')

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()

    def tokenize_and_segment(self, text):
        '''
        :param text: one sentence,
        E.G., text = "Here is the sentence I want embeddings for."
        :return: tokens_tensor, segments_tensors
        '''
        marked_text = "[CLS] " + text + " [SEP]"

        # Tokenize a sentence with the BERT tokenizer.
        tokenized_text = self.tokenizer.tokenize(marked_text)

        # Fix the number of tokens to 512 (defined by BERT model)
        tokenized_text = tokenized_text[:511] + [tokenized_text[-1]]

        # Map the token strings to their vocabulary indeces.
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Mark each of the tokens as belonging to sentence "1".
        segments_ids = [1] * len(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        return tokens_tensor, segments_tensors

    def compute_sentence_embedding(self, text):
        tokens_tensor, segments_tensors = self.tokenize_and_segment(text)
        with torch.no_grad():
            # `encoded_layers` has shape [num bert layers (12) x batch size x len(tokens_tensor[0]) x 768]
            encoded_layers, _ = self.model(tokens_tensor, segments_tensors)

            # To get a single vector for a sentence we take last hiden layer of each token
            # and apply mean pooling producing a single 768 length vector.
            # `token_vecs` is a tensor with shape [len(tokens_tensor[0]) x 768]
            token_vecs = encoded_layers[11][0]

            # Calculate the average of all 22 token vectors.
            sentence_embedding = torch.mean(token_vecs, dim=0)

        return sentence_embedding


# # an example
# bert = BERT()
# text = "Here is the sentence I want embeddings for."
# sentence_embedding = bert.compute_sentence_embedding(text)
