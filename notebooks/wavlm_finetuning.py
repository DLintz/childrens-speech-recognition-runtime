"""
Fine-tuning WavLM Base+ para On Top of Pasketti - Phonetic Track
Fase 3 do cronograma - VERSÃO CORRIGIDA
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2Processor, 
    Wav2Vec2ForCTC, 
    Wav2Vec2CTCTokenizer,
    TrainingArguments,
    Trainer,
    Wav2Vec2FeatureExtractor
)
import librosa
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import evaluate

# ============================================
# CONFIGURAÇÕES
# ============================================
class Config:
    # Caminhos
    DATA_DIR = Path("../data/processed")
    AUDIO_DIR = DATA_DIR / "audio"
    METADATA_FILE = DATA_DIR / "metadata.csv"
    TRAIN_SPLIT = DATA_DIR / "train_split.csv"
    VAL_SPLIT = DATA_DIR / "val_split.csv"
    OUTPUT_DIR = Path("../models/wavlm_phonetic")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # Áudio
    TARGET_SR = 16000
    MAX_AUDIO_LEN = 10  # segundos
    
    # Modelo
    MODEL_NAME = "microsoft/wavlm-base-plus"
    FREEZE_FEATURE_ENCODER = False  # False = fine-tuning profundo
    
    # Treino
    BATCH_SIZE = 8  # Ajuste conforme sua GPU
    GRADIENT_ACCUMULATION = 4
    LEARNING_RATE = 3e-5
    WARMUP_STEPS = 500
    MAX_EPOCHS = 5
    EVAL_STEPS = 500
    SAVE_STEPS = 1000
    LOGGING_STEPS = 100
    
    # Vocabulário IPA (50 caracteres únicos identificados na EDA)
    IPA_VOCAB = [
        ' ', 'i', 'ɪ', 'ɑ', 'n', 'ə', 't', 'd', 's', 'k', 'ʌ', 'ʊ', 'w', 
        'æ', 'o', 'l', 'ɛ', 'm', 'b', 'ɹ', 'ŋ', 'p', 'ɡ', 'f', 'v', 'ð', 
        'θ', 'z', 'ʃ', 'ʒ', 'h', 'j', 'ɝ', 'ɑɪ', 'aɪ', 'ɔ', 'u', 'e', 'ɚ',
        'ɜ', 'ɟ', 'x', 'ʁ', 'ʝ', 'c', 'ç', 'ɬ', 'ʔ', 'ɡ', 'ɑʊ'
    ]
    
    @classmethod
    def create_vocab_dict(cls):
        return {char: i for i, char in enumerate(cls.IPA_VOCAB)}
    
    @classmethod
    def get_vocab_size(cls):
        return len(cls.IPA_VOCAB)


# ============================================
# DATASET PERSONALIZADO
# ============================================
class PhoneticDataset(Dataset):
    def __init__(self, metadata_df, audio_dir, processor, target_sr=16000, max_audio_len=10):
        self.metadata = metadata_df
        self.audio_dir = audio_dir
        self.processor = processor
        self.target_sr = target_sr
        self.max_audio_len = max_audio_len * target_sr
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # 1. Carregar áudio
        audio_path = self.audio_dir / f"{row['id']}.wav"
        if not audio_path.exists():
            audio_path = self.audio_dir / f"{row['original_id']}.wav"
        
        try:
            audio, sr = librosa.load(audio_path, sr=self.target_sr)
        except Exception as e:
            print(f"Erro ao carregar {audio_path}: {e}")
            audio = np.zeros(int(self.target_sr * 1.0))
        
        # 2. Truncar/pad
        if len(audio) > self.max_audio_len:
            audio = audio[:self.max_audio_len]
        else:
            padding = self.max_audio_len - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')
        
        # 3. Processar áudio
        input_values = self.processor(audio, sampling_rate=self.target_sr, return_tensors="pt").input_values[0]
        
        # 4. Processar transcrição
        text = row['phonetic_text']
        with self.processor.as_target_processor():
            labels = self.processor(text, return_tensors="pt").input_ids[0]
        
        return {
            "input_values": input_values,
            "labels": labels,
            "attention_mask": torch.ones_like(input_values)
        }

# ============================================
# DATA COLLATOR
# ============================================
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = "longest"
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )
        
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        
        return batch

# ============================================
# MÉTRICAS (CER)
# ============================================
def compute_metrics(pred, processor):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    
    pred_str = processor.batch_decode(pred_ids)
    
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, group_tokens=False)
    
    cer_metric = evaluate.load("cer")
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    
    return {"cer": cer}

# ============================================
# FUNÇÃO PRINCIPAL DE TREINO
# ============================================
def train_wavlm():
    print("="*60)
    print("🎯 FASE 3: FINE-TUNING WavLM BASE+")
    print("="*60)
    
    # 1. Carregar dados
    print("\n📂 Carregando splits...")
    train_df = pd.read_csv(Config.TRAIN_SPLIT)
    val_df = pd.read_csv(Config.VAL_SPLIT)
    print(f"   Treino: {len(train_df)} amostras")
    print(f"   Validação: {len(val_df)} amostras")
    
    # 2. Criar vocabulário
    print("\n🔤 Configurando tokenizer IPA...")
    vocab_dict = Config.create_vocab_dict()
    
    vocab_file = Config.OUTPUT_DIR / "vocab.json"
    with open(vocab_file, "w") as f:
        json.dump(vocab_dict, f)
    
    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_file),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token=" ",
        replace_word_delimiter_char=" "
    )
    
    # 3. Feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=Config.TARGET_SR,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False
    )
    
    # 4. Processor
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )
    
    # 5. Datasets
    print("\n📦 Criando datasets...")
    train_dataset = PhoneticDataset(
        train_df, Config.AUDIO_DIR, processor, 
        Config.TARGET_SR, Config.MAX_AUDIO_LEN
    )
    val_dataset = PhoneticDataset(
        val_df, Config.AUDIO_DIR, processor,
        Config.TARGET_SR, Config.MAX_AUDIO_LEN
    )
    
    # 6. Modelo
    print("\n🤖 Carregando WavLM Base+...")
    model = Wav2Vec2ForCTC.from_pretrained(
        Config.MODEL_NAME,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    
    # 7. Fine-tuning profundo
    if Config.FREEZE_FEATURE_ENCODER:
        model.freeze_feature_encoder()
        print("   🔒 Feature encoder congelado")
    else:
        print("   🔓 Fine-tuning profundo (tudo descongelado)")
    
    # 8. Training arguments (VERSÃO CORRIGIDA)
    training_args = TrainingArguments(
        output_dir=str(Config.OUTPUT_DIR),
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION,
        evaluation_strategy="steps",
        eval_steps=Config.EVAL_STEPS,
        save_steps=Config.SAVE_STEPS,
        logging_steps=Config.LOGGING_STEPS,
        learning_rate=Config.LEARNING_RATE,
        warmup_steps=Config.WARMUP_STEPS,
        num_train_epochs=Config.MAX_EPOCHS,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False
    )
    # 8. Training arguments (VERSÃO MÍNIMA - deve funcionar em qualquer versão)
    # 8. Training arguments (VERSÃO COMPATÍVEL)
    training_args = TrainingArguments(
        output_dir=str(Config.OUTPUT_DIR),
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        num_train_epochs=Config.MAX_EPOCHS,
        save_steps=Config.SAVE_STEPS,
        save_total_limit=3,
        logging_steps=Config.LOGGING_STEPS,
        remove_unused_columns=False
    )
    
    # Adicionar evaluation apenas se a versão suportar
    if hasattr(training_args, 'evaluation_strategy'):
        training_args.evaluation_strategy = "steps"
        training_args.eval_steps = Config.EVAL_STEPS
    
    # Adicionar outros parâmetros opcionais se existirem
    if hasattr(training_args, 'gradient_accumulation_steps'):
        training_args.gradient_accumulation_steps = Config.GRADIENT_ACCUMULATION
    
    if hasattr(training_args, 'warmup_steps'):
        training_args.warmup_steps = Config.WARMUP_STEPS
    
    if hasattr(training_args, 'fp16'):
        training_args.fp16 = torch.cuda.is_available()
    
    if hasattr(training_args, 'load_best_model_at_end'):
        training_args.load_best_model_at_end = True
        training_args.metric_for_best_model = "cer"
        training_args.greater_is_better = False
    
    if hasattr(training_args, 'report_to'):
        training_args.report_to = "none"
    
    if hasattr(training_args, 'dataloader_num_workers'):
        training_args.dataloader_num_workers = 2
        
    # 9. Data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")
    
    # 10. Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.feature_extractor,
    )
    
    # 11. Treinar
    print("\n" + "="*60)
    print("🚀 INICIANDO TREINAMENTO")
    print("="*60)
    print(f"""
    Configurações:
    • Batch size: {Config.BATCH_SIZE} (efetivo: {Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION})
    • Learning rate: {Config.LEARNING_RATE}
    • Épocas: {Config.MAX_EPOCHS}
    • Fine-tuning profundo: {not Config.FREEZE_FEATURE_ENCODER}
    """)
    
    trainer.train()
    
    # 12. Salvar
    print("\n💾 Salvando modelo final...")
    trainer.save_model(str(Config.OUTPUT_DIR / "final_model"))
    processor.save_pretrained(str(Config.OUTPUT_DIR / "final_model"))
    
    print("\n" + "="*60)
    print("✅ FASE 3 CONCLUÍDA!")
    print("="*60)
    print(f"""
    📁 Modelo salvo em: {Config.OUTPUT_DIR}/final_model/
    """)
    
    return trainer, model, processor

# ============================================
# INFERÊNCIA
# ============================================
def test_inference(model, processor, audio_path):
    audio, _ = librosa.load(audio_path, sr=Config.TARGET_SR)
    inputs = processor(audio, sampling_rate=Config.TARGET_SR, return_tensors="pt")
    
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    return transcription

if __name__ == "__main__":
    trainer, model, processor = train_wavlm()