import math
import torch


class ConvLSTMEnc(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        latent_size: int,
        vocab_size: int = None,
        max_in_len: int = 256,
        letter_chunk: int = 8,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.vocab_size = vocab_size
        self.max_in_len = max_in_len

        if self.vocab_size:
            self.matrix = torch.nn.Embedding(
                num_embeddings=vocab_size, embedding_dim=input_size
            )
        else:
            self.matrix = None
        self.conv_kernel = input_size * letter_chunk
        self.conv_size = (max_in_len // letter_chunk) // 2

        self.conv = torch.nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.conv_kernel,
            stride=self.conv_kernel,
        )
        self.avg_pool = torch.nn.AvgPool1d(kernel_size=2, stride=2)
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=2,
            dropout=0.1,
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_size, out_features=hidden_size // 3 * 4),
            torch.nn.Linear(
                in_features=hidden_size // 3 * 4, out_features=hidden_size // 2
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=hidden_size // 2,
                out_features=self.latent_size - self.conv_size,
            ),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):

        if self.matrix:
            x = self.matrix(x)
        elif len(x.shape) == 2:
            x = x.unsqueeze(-1)
        x = x.to(torch.float32)
        rnn_in = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        h, (hn, cn) = self.lstm(rnn_in)
        fc_in, _ = torch.nn.utils.rnn.pad_packed_sequence(
            h, batch_first=True, total_length=self.max_in_len
        )
        fc_in = torch.sum(fc_in, dim=1) / lengths.unsqueeze(-1).to(fc_in.device)
        fc_out = self.fc(fc_in).squeeze(0).unsqueeze(1)
        conv_in = x.reshape(x.size(0), 1, -1)
        conv_out = self.conv(conv_in)
        avg_out = self.avg_pool(conv_out)
        out = torch.cat([avg_out, fc_out], dim=-1)

        return out


class ConvLSTMDec(torch.nn.Module):
    def __init__(
        self, latent_size: int, hidden_size: int, vocab_size: int, max_in_len: int = 256
    ):
        super().__init__()

        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_in_len = max_in_len

        self.lstm = torch.nn.LSTM(
            input_size=latent_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=2,
            dropout=0,
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_size, out_features=hidden_size // 3 * 4),
            torch.nn.Linear(
                in_features=hidden_size // 3 * 4, out_features=hidden_size // 2
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_size // 2, out_features=vocab_size),
        )

    def forward(self, z: torch.Tensor, lengths: torch.Tensor):
        z = z * math.sqrt(self.latent_size)
        z_expanded = z.repeat(1, self.max_in_len, 1)
        rnn_in = torch.nn.utils.rnn.pack_padded_sequence(
            z_expanded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        h, _ = self.lstm(rnn_in)
        fc_in, _ = torch.nn.utils.rnn.pad_packed_sequence(
            h, batch_first=True, total_length=self.max_in_len
        )
        out = self.fc(fc_in)
        return out


class ConvLSTM(torch.nn.Module):
    def __init__(
        self,
        embed_size: int,
        hidden_size_enc: int,
        hidden_size_dec: int,
        latent_size: int,
        vocab_size: int,
        use_embed_matrix: bool = False,
        max_in_len: int = 256,
        letter_chunk: int = 8,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size_enc = hidden_size_enc
        self.hidden_size_dec = hidden_size_dec
        self.latent_size = latent_size
        self.vocab_size = vocab_size
        self.use_embed_matrix = use_embed_matrix

        if use_embed_matrix:
            enc_vocab = vocab_size
        else:
            enc_vocab = None
        self.enc = ConvLSTMEnc(
            input_size=embed_size,
            hidden_size=hidden_size_enc,
            latent_size=latent_size,
            vocab_size=enc_vocab,
            max_in_len=max_in_len,
            letter_chunk=letter_chunk,
        )
        self.dec = ConvLSTMDec(
            latent_size=latent_size,
            hidden_size=hidden_size_dec,
            vocab_size=vocab_size,
            max_in_len=max_in_len,
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        z = self.enc(x, lengths)
        out = self.dec(z, lengths)
        return out

    def get_z(self, x: torch.Tensor, lengths: torch.Tensor):
        z = self.enc(x, lengths)
        return z


class ConvEmbedder(torch.nn.Module):
    def __init__(
        self,
        embed_size: int,
        hidden_size_enc: int,
        hidden_size_dec: int,
        latent_size: int,
        vocab_size: int,
        use_embed_matrix: bool = False,
        max_in_len: int = 256,
        letter_chunk: int = 8,
    ):
        super().__init__()
        self.conv_lstm = ConvLSTM(
            embed_size=embed_size,
            hidden_size_enc=hidden_size_enc,
            hidden_size_dec=hidden_size_dec,
            latent_size=latent_size,
            vocab_size=vocab_size,
            use_embed_matrix=use_embed_matrix,
            max_in_len=max_in_len,
            letter_chunk=letter_chunk,
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        b, t, f = x.shape
        x_re = x.view(b * t, -1)
        lengths_re = lengths.view(b * t)
        encoded = self.conv_lstm.get_z(x_re, lengths_re)
        encoded = encoded.view(b, t, -1)
        return encoded


class PosEncoder(torch.nn.Module):
    def __init__(self, d_model: int = 128, max_len: int = 300):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        positions = torch.arange(max_len).unsqueeze(1)
        div_factor = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_factor)
        pe[:, 1::2] = torch.cos(positions * div_factor)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        return x + self.pe[:, : x.size(1), :]

class PredTransformer(torch.nn.Module):
    def __init__(self, d_model: int = 512, n_head: int = 8, dec_layer: int = 6,
                 enc_layer: int = 6, dim_forward=1024):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dec_layer = dec_layer
        self.enc_layer = enc_layer
        self.dim_forward = dim_forward

        self.pe = PosEncoder(d_model=d_model, max_len=300)
        self.transformer = torch.nn.Transformer(d_model=d_model, batch_first=True, nhead=n_head,
                                                num_decoder_layers=dec_layer, num_encoder_layers=enc_layer,
                                                dim_feedforward=dim_forward)

    def forward(self, z: torch.Tensor, tgt_z: torch.Tensor, masks: torch.Tensor):
        src = z * math.sqrt(self.d_model)
        src = self.pe(src)
        tgt = tgt_z * math.sqrt(self.d_model)
        tgt = self.pe(tgt)
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt.size(1),
                                                                        device=tgt.device)
        tgt_padding_masks = torch.cat([torch.zeros(masks.size(0), 1, device=masks.device), masks[:, :-1]], dim=1)
        out = self.transformer(src=src, tgt=tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_masks,
                               src_key_padding_mask=masks)

        return out

class TaggedTransformer(torch.nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        dim_forward: int = 1024,
        n_head: int = 2,
        num_layers: int = 2,
        num_class: int = 5,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_forward = dim_forward
        self.n_head = n_head
        self.num_layers = num_layers
        self.num_class = num_class

        self.pe = PosEncoder(d_model=d_model, max_len=300)

        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_head,
            dim_feedforward=self.dim_forward,
            batch_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=d_model, out_features=d_model // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=d_model // 2, out_features=self.num_class),
        )

    def forward(self, z: torch.Tensor, masks: torch.Tensor):
        z = z * math.sqrt(self.d_model)
        z = self.pe(z)
        z = self.transformer(z, src_key_padding_mask=masks)
        z = self.fc(z)
        return z
