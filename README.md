# advanced-audio-processing
[project report and documentation](https://tuni-my.sharepoint.com/:w:/r/personal/minh_2_nguyen_tuni_fi/_layouts/15/Doc.aspx?sourcedoc=%7B3DF43AB1-761A-459C-93FD-7A48174A85AF%7D&file=Document.docx&action=default&mobileredirect=true)

## structure
```
biencoders/
├── dataset/
│   ├── aggregated.py
│   ├── audiocaps.py
│   ├── clothov2.py
│   ├── utils.py
│   └── wavcaps.py
├── model/
│   ├── biencoder.py
│   └── utils.py
├── tests/
└── utils/
helpers/
├── preprocess_audiocaps.py
├── preprocess_clothov2.py
└── preprocess_wavcaps.py
LICENSE
predict.py
README.md
requirements.txt
setup.py
train.py
utils/
```
## Run instructions

### install depenencies

```
> pip install -r requirements.txt
```

### Download dataset

```bash
> python helpers/audiocaps_downloader.py [--train-size 10] [--val-size 5] [--test-size 5]
```

### Run training

```bash
> python train.py
```

## Train logs

```bash
2025-03-14 01:09:32.786351: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1741914572.807962   15788 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1741914572.814622   15788 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2025-03-14 01:09:32.839252: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Process on cuda

[INFO] aggregated.py | Initializing AudioCaps dataset for split train...
[WARNING] aggregated.py | Clotho dataset not provided. Skipping...
[INFO] aggregated.py | Initializing AudioCaps dataset for split val...
[WARNING] aggregated.py | Clotho dataset not provided. Skipping...
[INFO] aggregated.py | Initializing AudioCaps dataset for split test...
[WARNING] aggregated.py | Clotho dataset not provided. Skipping...
config.json: 100% 1.60k/1.60k [00:00<00:00, 10.3MB/s]
model.safetensors: 100% 378M/378M [00:01<00:00, 246MB/s]
Some weights of Wav2Vec2Model were not initialized from the model checkpoint at facebook/wav2vec2-base-960h and are newly initialized: ['wav2vec2.masked_spec_embed']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
config.json: 100% 481/481 [00:00<00:00, 3.14MB/s]
model.safetensors: 100% 499M/499M [00:05<00:00, 98.6MB/s]
Some weights of RobertaModel were not initialized from the model checkpoint at roberta-base and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Total Parameters: 219,607,936
Trainable Parameters: 219,607,936
Epoch 1/100: 100% 63/63 [00:51<00:00,  1.22it/s]
Epoch 1/100, Loss: 8.220398479037815
Evaluating: 100% 13/13 [00:03<00:00,  3.57it/s]
Validation Cosine Similarity: 0.24336516857147217
Epoch 2/100: 100% 63/63 [00:53<00:00,  1.18it/s]
Epoch 2/100, Loss: 6.658346395643931
Evaluating: 100% 13/13 [00:03<00:00,  3.46it/s]
Validation Cosine Similarity: 0.26467788219451904
Epoch 3/100: 100% 63/63 [00:52<00:00,  1.21it/s]
Epoch 3/100, Loss: 5.828297914020599
Evaluating: 100% 13/13 [00:03<00:00,  3.52it/s]
Validation Cosine Similarity: 0.27323493361473083
Epoch 4/100: 100% 63/63 [00:51<00:00,  1.21it/s]
Epoch 4/100, Loss: 5.446388520891705
Evaluating: 100% 13/13 [00:03<00:00,  3.49it/s]
Validation Cosine Similarity: 0.2751373052597046
Epoch 5/100: 100% 63/63 [00:52<00:00,  1.20it/s]
Epoch 5/100, Loss: 4.969062192099435
Evaluating: 100% 13/13 [00:03<00:00,  3.32it/s]
Validation Cosine Similarity: 0.28172481060028076
Epoch 6/100: 100% 63/63 [00:51<00:00,  1.21it/s]
Epoch 6/100, Loss: 4.608618085346524
Evaluating: 100% 13/13 [00:04<00:00,  3.14it/s]
Validation Cosine Similarity: 0.28004613518714905
Epoch 7/100: 100% 63/63 [00:51<00:00,  1.23it/s]
Epoch 7/100, Loss: 4.4356515104808505
Evaluating: 100% 13/13 [00:03<00:00,  3.49it/s]
Validation Cosine Similarity: 0.28310295939445496
Epoch 8/100: 100% 63/63 [00:51<00:00,  1.23it/s]
Epoch 8/100, Loss: 4.0765482849544945
Evaluating: 100% 13/13 [00:03<00:00,  3.53it/s]
Validation Cosine Similarity: 0.2874350845813751
Epoch 9/100: 100% 63/63 [00:51<00:00,  1.21it/s]
Epoch 9/100, Loss: 3.874689189214555
Evaluating: 100% 13/13 [00:04<00:00,  3.21it/s]
Validation Cosine Similarity: 0.29378214478492737
```
