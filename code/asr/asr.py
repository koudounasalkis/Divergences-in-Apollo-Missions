from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor,WhisperTokenizer
from datasets import load_dataset, concatenate_datasets
import torch
from tqdm import tqdm
import pickle
import evaluate
import pandas as pd 
import librosa

STARTING_STEP = 0
MODEL = "openai/whisper-large-v3"
RESET_METRICS = True
SAVE_TO = "results-large/"
SAVE_EVERY = 1000
GRADIENT_ACCUMULATION_STEPS = 16
EVALUATION_SIZE = 100
LEARNING_RATE = 1e-4
EPOCHS = 3
BATCH_SIZE = 32
N = 10000


## Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

## Load the model, feature extractor, and tokenizer
model = WhisperForConditionalGeneration.from_pretrained(MODEL)
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL)
tokenizer = WhisperTokenizer.from_pretrained(MODEL)

## Load the dataset
dataset = pd.read_csv('fearless_steps_apollo/ASR_track2_train.csv')
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
print("Train dataset: ", len(dataset))

val_dataset = pd.read_csv('fearless_steps_apollo/ASR_track2_dev.csv')
print("Validation dataset: ", len(val_dataset))

## Freeze the encoder and unfreeze the decoder
model.get_encoder().requires_grad_(False)
model.get_decoder().requires_grad_(True)
model.proj_out.requires_grad_(True)

## Define the preprocessing function
def preprocess(data):
    processed_data = {}
    processed_data['input_features'] = feature_extractor(
        data['audio'], 
        sampling_rate=16_000, 
        return_tensors="pt"
        )['input_features'].to(device)
    processed_data['decoder_input_ids'] = tokenizer(
        '<|startoftranscript|>' + data['text'],
        return_tensors='pt'
        ).input_ids.to(device)
    processed_data['labels'] = tokenizer(
        data['text'] + '<|endoftext|>',
        return_tensors='pt'
        ).input_ids.to(device)
    
    return processed_data

## Define the evaluation function
wer_metric = evaluate.load("wer")

def evaluate(model, dataset):
    
    print("Evaluating...")
    
    with torch.no_grad():
        
        wer_scores = []
        losses = []
        
        # for item in tqdm(dataset):
        for index in tqdm(range(len(dataset))):

            item = dataset.loc[index]
            
            audio_id = item['id']
            audio = 'fearless_steps_apollo/FSC_P3_Train_Dev/Audio/Segments/ASR_track2/Dev/' + audio_id + '.wav'
            audio, _ = librosa.load(audio, sr=16_000)
            item['audio'] = audio

            input_features = feature_extractor(
                audio, 
                sampling_rate=16_000, 
                return_tensors="pt"
                )['input_features'].to(device)
            generated_ids = model.generate(
                input_features, 
                decoder_input_ids=tokenizer(
                    '<|startoftranscript|>', 
                    return_tensors='pt'
                    ).input_ids.to(device),
                max_length=100
                )
            generated_test_text = tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True
                )[0]
            wer_scores.append(wer_metric.compute(
                predictions=[generated_test_text.lower()], 
                references=[item['text'].lower()])
                )

            loss = model(**preprocess(item)).loss.item()
            losses.append(loss)

    return sum(losses)/len(losses), sum(wer_scores)/len(wer_scores)
        

## Load the metrics
if RESET_METRICS:
    with open(f'{SAVE_TO}losses.pkl','wb') as f:
        losses = []
        pickle.dump(losses,f)

    with open(f'{SAVE_TO}val_losses.pkl','wb') as  f:
        val_losses = []
        pickle.dump(val_losses,f)

    with open(f'{SAVE_TO}val_wers.pkl','wb') as f:
        val_wers = []
        pickle.dump(val_wers,f)


with open(f'{SAVE_TO}losses.pkl','rb') as f:
    losses = pickle.load(f)

with open(f'{SAVE_TO}val_losses.pkl','rb') as  f:
    val_losses = pickle.load(f)

with open(f'{SAVE_TO}val_wers.pkl','rb') as f:
    val_wers = pickle.load(f)


## Train the model
model.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

steps, gradient_steps, loss = 0, 0, 0
val_loss, val_wer = evaluate(
    model, 
    val_dataset.sample(frac=1, random_state=42).reset_index(drop=True)[:EVALUATION_SIZE]
    )

for epoch in range(EPOCHS):

    print("Epoch: ", epoch+1)

    for index in tqdm(range(0,len(dataset))):
        
        training_loop = tqdm(total=len(dataset))

        item = dataset.loc[index]
        audio_id = item['id']
        audio = 'fearless_steps_apollo/FSC_P3_Train_Dev/Audio/Segments/ASR_track2/Train/' + audio_id + '.wav'
        audio, _ = librosa.load(audio, sr=16_000)
        item['audio'] = audio

        loss = model(**preprocess(item)).loss
        loss.backward()
        losses.append(loss.item())
        steps += 1

        if steps % GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            gradient_steps += 1
        
        if steps % SAVE_EVERY == 0:
            
            val_loss,val_wer = evaluate(
                model,
                val_dataset.sample(frac=1, random_state=42).reset_index(drop=True)[:EVALUATION_SIZE])
            val_losses.append(val_loss)
            val_wers.append(val_wer)
            
            model.save_pretrained(f'{SAVE_TO}whisper-{steps}')

            with open(f'{SAVE_TO}losses.pkl','wb') as f:
                pickle.dump(losses,f)

            with open(f'{SAVE_TO}val_losses.pkl','wb') as f:
                pickle.dump(val_losses,f)

            with open(f'{SAVE_TO}val_wers.pkl','wb') as f:
                pickle.dump(val_wers,f)

        training_loop.set_description(f"Steps: {steps}   Gradient Updates: {gradient_steps}   Loss: {loss.item():.4f}   Val Loss: {val_loss:.4f}   Val WER: {val_wer:.4f}")