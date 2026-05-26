import argparse
import os
from data import CharVocab, DummyLogDataSet
from models import ConvEmbedder, TaggedTransformer
import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--job", required=True, type=str)
    parser.add_argument("-p","--path",required=True,type=str)
    args = parser.parse_args()

    job_name = args.job
    action_path = args.path
    if not action_path:
        print("No action path found")
        exit(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autoencoder_name = "ConvLSTM-E-32-H-196-L-128.pt"
    char_vocab = CharVocab()
    embed_size = 32
    hidden_size_enc = 196
    hidden_size_dec = 384
    latent_size = 128
    vocab_size = len(char_vocab)
    use_embed_matrix = True
    max_in_len = 200
    letter_chunk = 4

    embedder = ConvEmbedder(
        embed_size=embed_size,
        hidden_size_enc=hidden_size_enc,
        hidden_size_dec=hidden_size_dec,
        latent_size=latent_size,
        letter_chunk=letter_chunk,
        max_in_len=max_in_len,
        use_embed_matrix=use_embed_matrix,
        vocab_size=vocab_size,
    )

    try:
        embedder.conv_lstm.load_state_dict(
            torch.load(os.path.join(action_path, autoencoder_name), weights_only=True,map_location=torch.device(device))
        )
    except Exception as e:
        print("Can not load ConvLstmEncoder", autoencoder_name)
        print(e)
        exit(-1)

    enc_layer = 2
    n_head = 2
    dim_forward = 1024
    d_model = 128
    transformer_name = "TaggedTransformer-E-2-H-2-F-1024-D-128.pt"
    model = TaggedTransformer(
        d_model=d_model,
        dim_forward=dim_forward,
        n_head=n_head,
        num_layers=enc_layer,
        num_class=5,
    )

    try:
        model.load_state_dict(
            torch.load(os.path.join(action_path, transformer_name), weights_only=True,map_location=torch.device(device))
        )
    except Exception as e:
        print("Can not load Transformer", transformer_name)
        print(e)
        exit(-1)

    model.eval()
    step = 15
    frame_size = 15
    batch_size = 1
    dataset = DummyLogDataSet(step=step, frame_size=frame_size, pad_tag=6)
    dataset.add_from_file(f"{job_name}.txt")
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    tag_names = ["Flaky test","Configuration drift","Security anomaly","Silent failure"]
    anomalies_present = [False for _ in range(len(tag_names))]

    for i, (data, lengths, masks, tags) in enumerate(data_loader):
        with torch.no_grad():
            z = embedder(data, lengths)
            out = model(z, masks)
            prediction = torch.argmax(torch.softmax(out, dim=-1), dim=-1)
            prediction = prediction.view(-1)
            tags = tags.view(-1)
            prediction = prediction[tags != dataset.pad_tag]
            log_str = dataset.get_str_log_item(i)
            if prediction.max().item()> 0:
                for j in range(len(log_str)):
                    if prediction[j]>0:
                        anomalies_present[prediction[j]-1] = True
                        print(f"{tag_names[prediction[j]-1]:<20}| {log_str[j]}",end="")
                    else:
                        print(f"{' '*20}| {log_str[j]}",end="")
    if True in anomalies_present:
        with open("body.html","w",encoding="utf-8") as body:
            body.write(f"<h2>Log anomaly detection bot has found some anomalies in job: {job_name}</h2>")
            anomalies_str = ''
            for i in range(len(anomalies_present)):
                if anomalies_present[i]:
                    anomalies_str += f"{tag_names[i]},"
            anomalies_str = anomalies_str[:-1]
            body.write(f"<p>Found {anomalies_str} view attached file for details</p>")
