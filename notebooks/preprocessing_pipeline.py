"""
Pipeline de Pré-processamento e Data Augmentation
Para On Top of Pasketti - Phonetic Track

Features:
- Resample de alta qualidade para 16kHz (kaiser_best)
- Data augmentation específico para fonemas raros
- Tratamento balanceado para idade 8-11 (Transfer Learning + Oversampling moderado)
- Salvamento em formato otimizado para treino
"""

import os
import json
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import random
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURAÇÕES
# ============================================
class Config:
    # Caminhos
    DATA_DIR = Path("../data/")
    RAW_AUDIO_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    TRANSCRIPT_FILE = DATA_DIR / "train_phon_transcripts.jsonl"
    METADATA_FILE = PROCESSED_DIR / "metadata.csv"
    
    # Áudio
    TARGET_SR = 16000  # Taxa para modelos WavLM/XLS-R
    RESAMPLE_TYPE = 'kaiser_best'  # Qualidade máxima
    
    # Data Augmentation para fonemas raros
    FONEMAS_RAROS = ['ɟ', 'x', 'ʁ', 'ʝ', 'c', 'ç', 'ʒ', 'ɬ']  # <50 ocorrências
    AUGMENT_FACTOR = 3  # Triplicar amostras raras
    
    # Aumentos
    PITCH_SHIFTS = [-2, -1, 1, 2]  # Semitons
    SPEED_FACTORS = [0.9, 1.1]  # 90% e 110% da velocidade
    NOISE_SNR = 20  # dB para ruído adicionado
    
    # Idade 8-11 (oversampling moderado)
    IDADE_RARA = '8-11'
    OVERSAMPLE_FACTOR = 3  # Triplicar amostras desta idade
    
    # Processamento
    N_WORKERS = 4  # Threads para processamento paralelo
    CHUNK_SIZE = 100  # Salvar em lotes para não sobrecarregar memória
    
    @classmethod
    def create_dirs(cls):
        cls.PROCESSED_DIR.mkdir(exist_ok=True)
        (cls.PROCESSED_DIR / "audio").mkdir(exist_ok=True)

# ============================================
# 1. CARREGAR E ANALISAR DADOS
# ============================================
class DataLoader:
    @staticmethod
    def load_transcriptions():
        """Carrega arquivo JSONL de transcrições"""
        data = []
        with open(Config.TRANSCRIPT_FILE, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        df = pd.DataFrame(data)
        
        # Garantir tipos corretos
        df['utterance_id'] = df['utterance_id'].astype(str)
        df['age_bucket'] = df['age_bucket'].astype(str)
        df['phonetic_text'] = df['phonetic_text'].astype(str)
        
        return df
    
    @staticmethod
    def analyze_fonemas_raros(df):
        """Identifica quais IDs contêm fonemas raros"""
        ids_com_fonema_raros = {f: [] for f in Config.FONEMAS_RAROS}
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Analisando fonemas raros"):
            texto = row['phonetic_text']
            for fonema in Config.FONEMAS_RAROS:
                if fonema in texto:
                    ids_com_fonema_raros[fonema].append(row['utterance_id'])
        
        return ids_com_fonema_raros

# ============================================
# 2. PROCESSAMENTO DE ÁUDIO (RESAMPLE)
# ============================================
class AudioProcessor:
    @staticmethod
    def resample_audio(audio_path, target_sr=Config.TARGET_SR):
        """
        Resample de alta qualidade usando kaiser_best
        Retorna: (audio_resampled, sr_original, sr_target)
        """
        try:
            # Carregar com taxa original
            y, orig_sr = librosa.load(audio_path, sr=None)
            
            # Resample se necessário
            if orig_sr != target_sr:
                # kaiser_best é o método de mais alta qualidade
                y = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr, 
                                    res_type=Config.RESAMPLE_TYPE)
            
            return y, orig_sr, target_sr
        except Exception as e:
            print(f"Erro em {audio_path}: {e}")
            return None, None, None
    
    @staticmethod
    def process_and_save(audio_id, audio_path, output_dir):
        """Processa um único áudio e salva versão resampleada"""
        output_path = output_dir / f"{audio_id}.wav"  # WAV para carregamento rápido
        
        # Se já existe, não processar de novo
        if output_path.exists():
            return str(output_path)
        
        y, orig_sr, target_sr = AudioProcessor.resample_audio(audio_path)
        if y is not None:
            # Salvar como WAV (formato padrão para treino)
            sf.write(output_path, y, target_sr)
            return str(output_path)
        return None

# ============================================
# 3. DATA AUGMENTATION
# ============================================
class AudioAugmenter:
    @staticmethod
    def add_noise(y, sr, snr_db=Config.NOISE_SNR):
        """Adiciona ruído branco com SNR específica"""
        # Calcular potência do sinal
        signal_power = np.mean(y**2)
        
        # Calcular potência do ruído para SNR desejada
        noise_power = signal_power / (10**(snr_db/10))
        
        # Gerar ruído
        noise = np.random.normal(0, np.sqrt(noise_power), len(y))
        
        return y + noise
    
    @staticmethod
    def pitch_shift(y, sr, n_steps):
        """Muda o tom (pitch) em semitons"""
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    
    @staticmethod
    def time_stretch(y, speed_factor):
        """Muda velocidade sem alterar pitch"""
        return librosa.effects.time_stretch(y, rate=speed_factor)
    
    @staticmethod
    def generate_augmented_samples(y, sr, augment_type='all'):
        """
        Gera múltiplas versões aumentadas do mesmo áudio
        augment_type: 'all', 'pitch', 'speed', 'noise'
        """
        augmented = []
        
        # Variações de pitch
        if augment_type in ['all', 'pitch']:
            for shift in Config.PITCH_SHIFTS:
                try:
                    y_pitch = AudioAugmenter.pitch_shift(y, sr, shift)
                    augmented.append((y_pitch, f"pitch_{shift}"))
                except:
                    pass
        
        # Variações de velocidade
        if augment_type in ['all', 'speed']:
            for factor in Config.SPEED_FACTORS:
                try:
                    y_speed = AudioAugmenter.time_stretch(y, factor)
                    augmented.append((y_speed, f"speed_{factor}"))
                except:
                    pass
        
        # Ruído
        if augment_type in ['all', 'noise']:
            try:
                y_noise = AudioAugmenter.add_noise(y, sr)
                augmented.append((y_noise, "noise"))
            except:
                pass
        
        return augmented

# ============================================
# 4. PIPELINE PRINCIPAL
# ============================================
class PreprocessingPipeline:
    def __init__(self):
        Config.create_dirs()
        self.processed_audio_dir = Config.PROCESSED_DIR / "audio"
        self.processed_audio_dir.mkdir(exist_ok=True)
        
        # Carregar dados
        print("📂 Carregando transcrições...")
        self.df = DataLoader.load_transcriptions()
        print(f"   Total: {len(self.df)} amostras")
        
        # Identificar amostras especiais
        print("🔍 Identificando amostras para augmentation...")
        self.fonemas_raros_ids = DataLoader.analyze_fonemas_raros(self.df)
        
        # IDs para oversampling (idade 8-11)
        self.idade_rara_ids = self.df[self.df['age_bucket'] == Config.IDADE_RARA]['utterance_id'].tolist()
        print(f"   • {len(self.idade_rara_ids)} amostras idade {Config.IDADE_RARA}")
        
        # Total de IDs únicos com fonemas raros
        todos_fonemas_raros = set()
        for ids in self.fonemas_raros_ids.values():
            todos_fonemas_raros.update(ids)
        print(f"   • {len(todos_fonemas_raros)} amostras com fonemas raros")
    
    def process_audio_file(self, audio_id, augment=False, augment_type='all'):
        """Processa um único arquivo de áudio, com opção de augmentation"""
        audio_path = Config.RAW_AUDIO_DIR / f"{audio_id}.flac"
        if not audio_path.exists():
            return []
        
        # 1. Resample e salvar versão base
        base_path = self.processed_audio_dir / f"{audio_id}.wav"
        
        # Se não existe versão base, criar
        if not base_path.exists():
            y, _, _ = AudioProcessor.resample_audio(audio_path)
            if y is not None:
                sf.write(base_path, y, Config.TARGET_SR)
        
        # Se não precisa de augmentation, retorna só o base
        if not augment:
            return [{'id': audio_id, 'path': str(base_path), 'augmented': False}]
        
        # 2. Carregar para augmentation
        y, _ = librosa.load(base_path, sr=Config.TARGET_SR)
        
        # 3. Gerar versões aumentadas
        augmented_samples = AudioAugmenter.generate_augmented_samples(y, Config.TARGET_SR, augment_type)
        
        results = [{'id': audio_id, 'path': str(base_path), 'augmented': False}]
        
        for i, (y_aug, aug_name) in enumerate(augmented_samples):
            aug_path = self.processed_audio_dir / f"{audio_id}_{aug_name}.wav"
            sf.write(aug_path, y_aug, Config.TARGET_SR)
            results.append({
                'id': f"{audio_id}_{aug_name}",
                'original_id': audio_id,
                'path': str(aug_path),
                'augmented': True,
                'augmentation': aug_name
            })
        
        return results
    
    def create_augmentation_plan(self):
        """Cria plano de quais amostras aumentar e quanto"""
        plan = []
        
        # 1. Amostras com fonemas raros (aumentar)
        todos_fonemas_raros = set()
        for ids in self.fonemas_raros_ids.values():
            todos_fonemas_raros.update(ids)
        
        for audio_id in todos_fonemas_raros:
            plan.append({
                'audio_id': audio_id,
                'augment': True,
                'reason': 'fonema_raro',
                'augment_type': 'all',
                'count': Config.AUGMENT_FACTOR
            })
        
        # 2. Amostras de idade rara (Opção B moderada + Opção C)
        # Vamos triplicar mas com augmentation menos agressivo
        for audio_id in self.idade_rara_ids:
            if audio_id not in todos_fonemas_raros:  # Evitar duplicar
                plan.append({
                    'audio_id': audio_id,
                    'augment': True,
                    'reason': 'idade_rara',
                    'augment_type': 'pitch',  # Só pitch shift para não distorcer muito
                    'count': Config.OVERSAMPLE_FACTOR
                })
        
        return plan
    
    def run(self):
        """Executa pipeline completo"""
        print("\n" + "="*60)
        print("🚀 INICIANDO PIPELINE DE PRÉ-PROCESSAMENTO")
        print("="*60)
        
        # Criar plano de augmentation
        plan = self.create_augmentation_plan()
        print(f"\n📋 Plano de augmentation:")
        print(f"   • Amostras com fonemas raros: {len([p for p in plan if p['reason'] == 'fonema_raro'])}")
        print(f"   • Amostras idade {Config.IDADE_RARA}: {len([p for p in plan if p['reason'] == 'idade_rara'])}")
        
        # Processar TODAS as amostras (base + augmentation)
        todas_amostras = []
        plan_dict = {p['audio_id']: p for p in plan}
        
        print("\n🔄 Processando áudios...")
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processando"):
            audio_id = row['utterance_id']
            
            # Verificar se precisa augmentation
            if audio_id in plan_dict:
                p = plan_dict[audio_id]
                # Gerar múltiplas versões (count = fator de aumento)
                for _ in range(p['count']):
                    resultados = self.process_audio_file(audio_id, augment=True, augment_type=p['augment_type'])
                    todas_amostras.extend(resultados)
            else:
                # Apenas versão base
                resultados = self.process_audio_file(audio_id, augment=False)
                todas_amostras.extend(resultados)
        
        # Criar DataFrame final com metadados
        print("\n📊 Criando metadados finais...")
        df_final = []
        
        for amostra in todas_amostras:
            # Buscar transcrição original
            if 'original_id' in amostra:
                row_orig = self.df[self.df['utterance_id'] == amostra['original_id']].iloc[0]
            else:
                row_orig = self.df[self.df['utterance_id'] == amostra['id']].iloc[0]
            
            df_final.append({
                'id': amostra['id'],
                'original_id': amostra.get('original_id', amostra['id']),
                'audio_path': amostra['path'],
                'phonetic_text': row_orig['phonetic_text'],
                'age_bucket': row_orig['age_bucket'],
                'augmented': amostra.get('augmented', False),
                'augmentation_type': amostra.get('augmentation', 'none')
            })
        
        df_metadata = pd.DataFrame(df_final)
        
        # Salvar metadados
        print("\n💾 Salvando metadados...")
        df_metadata.to_csv(Config.METADATA_FILE, index=False)
        
        # Estatísticas finais
        print("\n" + "="*60)
        print("✅ PIPELINE CONCLUÍDO!")
        print("="*60)
        print(f"""
📈 ESTATÍSTICAS FINAIS:
   • Amostras originais: {len(self.df)}
   • Amostras após augmentation: {len(df_metadata)}
   • Fator de aumento total: {len(df_metadata)/len(self.df):.1f}x
   
   • Amostras aumentadas: {df_metadata['augmented'].sum()}
   • Amostras não aumentadas: {(~df_metadata['augmented']).sum()}
   
   • Distribuição por idade:
        {df_metadata['age_bucket'].value_counts().to_string()}
   
📁 Arquivos salvos:
   • Áudios processados: {self.processed_audio_dir}/
   • Metadados: {Config.METADATA_FILE}
        """)
        
        return df_metadata

# ============================================
# 5. FUNÇÕES DE UTILIDADE PARA TREINO
# ============================================
class DatasetUtils:
    @staticmethod
    def load_metadata():
        """Carrega metadados processados"""
        if not Config.METADATA_FILE.exists():
            raise FileNotFoundError("Execute o pipeline primeiro!")
        return pd.read_csv(Config.METADATA_FILE)
    
    @staticmethod
    def create_train_val_split(df, val_split=0.1, stratify_by='age_bucket'):
        """Cria split treino/validação estratificado"""
        from sklearn.model_selection import train_test_split
        
        # Usar IDs originais para evitar data leakage
        unique_ids = df['original_id'].unique()
        ids_train, ids_val = train_test_split(
            unique_ids, 
            test_size=val_split, 
            random_state=42,
            stratify=df.groupby('original_id')['age_bucket'].first().loc[unique_ids] if stratify_by else None
        )
        
        train_df = df[df['original_id'].isin(ids_train)]
        val_df = df[df['original_id'].isin(ids_val)]
        
        print(f"Split treino/validação:")
        print(f"  • Treino: {len(train_df)} amostras ({len(ids_train)} originais)")
        print(f"  • Validação: {len(val_df)} amostras ({len(ids_val)} originais)")
        
        return train_df, val_df
    
    @staticmethod
    def get_audio_and_transcript(row):
        """Carrega áudio e transcrição para treino"""
        audio_path = row['audio_path']
        y, sr = librosa.load(audio_path, sr=Config.TARGET_SR)
        text = row['phonetic_text']
        return y, text

# ============================================
# 6. EXECUÇÃO
# ============================================
if __name__ == "__main__":
    # Executar pipeline
    pipeline = PreprocessingPipeline()
    df_metadata = pipeline.run()
    
    print("\n" + "="*60)
    print("🎯 PRÓXIMOS PASSOS:")
    print("="*60)
    print("""
    1. Use DatasetUtils.load_metadata() para carregar dados processados
    2. Use DatasetUtils.create_train_val_split() para split treino/validação
    3. No treino, use get_audio_and_transcript() para carregar cada amostra
    
    Exemplo:
    
    from preprocessing_pipeline import DatasetUtils, Config
    
    # Carregar metadados
    df = DatasetUtils.load_metadata()
    
    # Split treino/validação
    train_df, val_df = DatasetUtils.create_train_val_split(df)
    
    # No loop de treino
    for _, row in train_df.iterrows():
        audio, text = DatasetUtils.get_audio_and_transcript(row)
        # ... seu código de treino ...
    """)