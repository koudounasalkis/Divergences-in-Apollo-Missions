from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor,WhisperTokenizer
from datasets import load_dataset
import torch
from tqdm import tqdm
from jiwer import wer
import evaluate
import pandas as pd 
import librosa

SIZE = -1
PROCESS_OUTPUT = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_to_test = [
    'results/whisper-base.en',
    'results/whisper-base',
    'results/whisper-small.en',
    'results/whisper-small',
    'results/whisper-medium.en',
    'results/whisper-medium',
    'results/whisper-large-v3',
    ]

## Load metrics
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

with torch.no_grad():

    for m, model_to_test in enumerate(models_to_test):

        print(f"Testing model: {model_to_test}")

        ## Load the dataset
        test_dataset = pd.read_csv('fearless_steps_apollo/ASR_track2_dev.csv')
        test_dataset = test_dataset.sample(frac=1, random_state=42).reset_index(drop=True)[:SIZE if SIZE != -1 else len(test_dataset)]
        print(f"Testing on {len(test_dataset)} samples")

        ## Load the model, feature extractor, and tokenizer
        feature_extractor = WhisperFeatureExtractor.from_pretrained(model_to_test)
        tokenizer = WhisperTokenizer.from_pretrained(model_to_test)
        decoder_input_ids = tokenizer("<|startoftranscript|>", return_tensors="pt").input_ids.to(device)
        model = WhisperForConditionalGeneration.from_pretrained(model_to_test)
        model.to(device)
        
        ## Set the model to evaluation mode
        model.eval()
        
        ## Initialize the scores
        wer_scores = []
        cer_scores = []

        ## Loop through the dataset
        for index in tqdm(range(len(test_dataset))):

            ## Load the audio
            audio_id = test_dataset.loc[index, 'id']
            audio = 'fearless_steps_apollo/FSC_P3_Train_Dev/Audio/Segments/ASR_track2/Dev/' + audio_id + '.wav'
            audio, _ = librosa.load(audio, sr=16_000)

            ## Extract the features
            input_features = feature_extractor(
                audio, 
                sampling_rate=16_000, 
                return_tensors="pt"
                )['input_features'].to(device)

            ## Generate the text
            generated_ids = model.generate(
                input_features, 
                decoder_input_ids=decoder_input_ids, 
                max_length=100
                )
            generated_test_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            if PROCESS_OUTPUT:
                generated_test_text = generated_test_text.replace('.', '')
                generated_test_text = generated_test_text.replace(',', '')
                generated_test_text = generated_test_text.replace('?', '')
                generated_test_text = generated_test_text.replace('!', '')
                generated_test_text = generated_test_text.lower()

            ## Add generated text to the dataset
            test_dataset.loc[index, 'generated_text'] = generated_test_text

            ## Compute the scores
            wer_scores.append(wer_metric.compute(
                predictions=[generated_test_text.lower()], 
                references=[test_dataset.loc[index, 'text'].lower()])
                )
            test_dataset.loc[index, 'wer'] = wer_scores[-1]
            cer_scores.append(cer_metric.compute(
                predictions=[generated_test_text.lower()], 
                references=[test_dataset.loc[index, 'text'].lower()])
                )
            test_dataset.loc[index, 'cer'] = cer_scores[-1]

        print(f"WER score for {model_to_test}: {100 * sum(wer_scores)/len(wer_scores)}")
        print(f"CER score for {model_to_test}: {100 * sum(cer_scores)/len(cer_scores)}")

        ## Save the dataset 
        name = model_to_test.replace('/', '_').replace('.', '_')
        test_dataset.to_csv(f'fearless_steps_apollo/ASR_track2_dev_{model_to_test}.csv', index=False)
