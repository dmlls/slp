"""Progressive Transformer for End-to-End SLP."""

import pytorch_lightning as pl
from omegaconf.dictconfig import DictConfig
from src.skelesign.constants import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, TARGET_PAD
from src.skelesign.batch import Batch
from src.skelesign.loss import RegLoss
from src.skelesign.vocabulary import Vocabulary
from .layers import EncoderLayer, DecoderLayer


class ProgressiveTransformer(pl.LightningModule):
    """Progressive Transformer for SLP."""

    def __init__(self,
                 src_embed: Embeddings,
                 trg_embed: Embeddings,
                 src_vocab: Vocabulary,
                 trg_vocab: Vocabulary,
                 config: DictConfig):
        super().__init__()
        self.learning_rate = config.training.learning_rate
        self.loss_function = RegLoss(config.training.loss,
                                     config.training.loss_scale)
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.bos_index = BOS_TOKEN
        self.pad_index = PAD_TOKEN
        self.eos_index = EOS_TOKEN
        self.target_pad = TARGET_PAD

        # Transformer Encoder
        enc_dropout = config.model.encoder.get("dropout", 0.) # Dropout
        enc_emb_dropout = config.model.encoder.embeddings.get("dropout", enc_dropout)
        self.encoder = TransformerEncoder(**config.model.encoder,
                                          emb_size=src_embed.embedding_dim,
                                          emb_dropout=enc_emb_dropout)

        # Transformer Decoder
        dec_dropout = config.model.decoder.get("dropout", 0.) # Dropout
        dec_emb_dropout = config.model.decoder.embeddings.get("dropout", dec_dropout)
        decoder_trg_trg = config.model.decoder.get("decoder_trg_trg", True)
        # Define target linear
        # Linear layer replaces an embedding layer - as this takes in the joints
        # size as opposed to a token
        trg_linear = nn.Linear(in_trg_size, cfg.decoder.embeddings.embedding_dim)
        self.decoder = TransformerDecoder(
            **config.model.decoder, encoder=self.encoder, vocab_size=len(trg_vocab),
            emb_size=trg_linear.out_features, emb_dropout=dec_emb_dropout,
            trg_size=out_trg_size, decoder_trg_trg_=decoder_trg_trg
        )

    # pylint: disable=arguments-differ
    def forward(
        self,
        src: Tensor,
        trg_input: Tensor,
        src_mask: Tensor,
        src_lengths: Tensor,
        trg_mask: Tensor = None,
        src_input: Tensor = None
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """Forward pass.

        First encodes the source sentence. Then produces the target one word at
        a time.

        :param src: source input
        :param trg_input: target input
        :param src_mask: source mask
        :param src_lengths: length of source inputs
        :param trg_mask: target mask
        :return: decoder outputs
        """

        # Encode the source sequence
        encoder_output, encoder_hidden = self.encode(src=src,
                                                     src_length=src_lengths,
                                                     src_mask=src_mask)
        unroll_steps = trg_input.size(1)

        # Add gaussian noise to the target inputs, if in training
        if (self.gaussian_noise) and (self.training) and (self.out_stds is not None):

            # Create a normal distribution of random numbers between 0-1
            noise = trg_input.data.new(trg_input.size()).normal_(0, 1)
            # Zero out the noise over the counter
            noise[:,:,-1] = torch.zeros_like(noise[:, :, -1])

            # Need to add a zero on the end of
            if self.future_prediction != 0:
                self.out_stds = torch.cat((self.out_stds,torch.zeros_like(self.out_stds)))[:trg_input.shape[-1]]

            # Need to multiply by the standard deviations
            noise = noise * self.out_stds

            # Add to trg_input multiplied by the noise rate
            trg_input = trg_input + self.noise_rate*noise

        # Decode the target sequence
        skel_out, dec_hidden, _, _ = self.decode(encoder_output=encoder_output,
                                                 src_mask=src_mask, trg_input=trg_input,
                                                 trg_mask=trg_mask)

        return skel_out

    def training_step(self, batch: Batch, batch_idx):
        """Perform a training step."""
        # First encode the batch, as this can be done in all one go
        encoder_output, encoder_hidden = self.encode(
            batch.src, batch.src_lengths, batch.src_mask)

        # Then decode the batch separately, as needs to be done iteratively
        # greedy decoding
        stacked_output, stacked_attention_scores = greedy(
                encoder_output=encoder_output,
                src_mask=batch.src_mask,
                embed=self.trg_embed,
                decoder=self.decoder,
                trg_input=batch.trg_input,
                model=self
        )

        loss = self.get_loss_for_batch(batch, self.loss_function)

        return {
            "stacked_output": stacked_output,
            "stacked_attention_scores": stacked_attention_scores,
            "loss": loss
        }

    def validation_step(self, batch, batch_idx):
        """Perform a validation step."""
        pass  # TODO

    def encode(
        self,
        src: Tensor,
        src_length: Tensor,
        src_mask: Tensor
    ) -> (Tensor, Tensor):
        """Encodes the source sentence.

        :param src:
        :param src_length:
        :param src_mask:
        :return: encoder outputs (output, hidden_concat)
        """
        # Encode an embedded source
        encode_output = self.encoder(self.src_embed(src), src_length, src_mask)
        return encode_output

    def decode(
        self,
        encoder_output: Tensor,
        src_mask: Tensor,
        trg_input: Tensor,
        trg_mask: Tensor = None
    ) -> (Tensor, Tensor, Tensor, Tensor):

        """Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param src_mask: source mask, 1 at valid tokens
        :param trg_input: target inputs
        :param unroll_steps: number of steps to unrol the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param trg_mask: mask for target steps
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """
        # Embed the target using a linear layer
        trg_embed = self.trg_embed(trg_input)
        # Apply decoder to the embedded target
        decoder_output = self.decoder(
            trg_embed=trg_embed, encoder_output=encoder_output,
            src_mask=src_mask,trg_mask=trg_mask)
        return decoder_output

    def get_loss_for_batch(
        self,
        batch: Batch,
        loss_function: nn.Module
    ) -> Tensor:
        """Compute non-normalized loss and number of tokens for a batch.

        :param batch: batch to compute loss for
        :param loss_function: loss function, computes for input and target
            a scalar loss for the complete batch
        :return: batch_loss: sum of losses over non-pad elements in the batch
        """
        # Forward through the batch input
        skel_out, _ = self.forward(
            src=batch.src, trg_input=batch.trg_input,
            src_mask=batch.src_mask, src_lengths=batch.src_lengths,
            trg_mask=batch.trg_mask
        )

        # Compute batch loss using skel_out and the batch target
        # Return batch loss = sum over all elements in batch that are not pad
        return loss_function(skel_out, batch.trg)

    def run_batch(self, batch: Batch, max_output_length: int,) -> (np.array, np.array):
        """Get outputs and attentions scores for a given batch

        :param batch: batch to generate hypotheses for
        :param max_output_length: maximum length of hypotheses
        :param beam_size: size of the beam for beam search, if 0 use greedy
        :param beam_alpha: alpha value for beam search
        :return: stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """
        # First encode the batch, as this can be done in all one go
        encoder_output, encoder_hidden = self.encode(
            batch.src, batch.src_lengths,
            batch.src_mask)

        # if maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = int(max(batch.src_lengths.cpu().numpy()) * 1.5)

        # Then decode the batch separately, as needs to be done iteratively
        # greedy decoding
        stacked_output, stacked_attention_scores = greedy(
                encoder_output=encoder_output,
                src_mask=batch.src_mask,
                embed=self.trg_embed,
                decoder=self.decoder,
                trg_input=batch.trg_input,
                model=self)

        return stacked_output, stacked_attention_scores

    def __repr__(self) -> str:
        """String representation.

        A description of encoder, decoder and embeddings.

        :return: string representation
        """
        return "%s(\n" \
               "\tencoder=%s,\n" \
               "\tdecoder=%s,\n" \
               "\tsrc_embed=%s,\n" \
               "\ttrg_embed=%s)" % (self.__class__.__name__, self.encoder,
                   self.decoder, self.src_embed, self.trg_embed)
