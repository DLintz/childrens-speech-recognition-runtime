"""
Pipeline de Pré-processamento e Data Augmentation - VERSÃO ROBUSTA
Para On Top of Pasketti - Phonetic Track
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
import warnings
warnings.filterwarnings('ignore')

# ============================================
# VERIFICAR DEPENDÊNCIAS
# ============================================
def check_dependencies():
    """Verifica se todas as dependências estão instaladas"""
    missing = []
    
    try:
        import resampy
        print("✅ resampy instalado")
    except ImportError:
        missing.append("resampy")
        print("❌ resampy faltando")
    
    try:
        import soundfile
        print("✅ soundfile instalado")
    except ImportError:
        missing.append("soundfile")
        print("❌ soundfile faltando")
    
    if missing:
        print(f"\n⚠️ Pacotes faltando: {missing}")
        print("Instale com: pip install " + " ".join(missing))
        return False
    return True

# ============================================
# CONFIGURAÇÕES
# ============================================
class Config:
    # Caminhos
    DATA_DIR = Path("../data/")
    RAW_AUDIO_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    PROCESSED_AUDIO_DIR = PROCESSED_DIR / "audio"
    TRANSCRIPT_FILE = DATA_DIR / "train_phon_transcripts.jsonl"
    METADATA_FILE = PROCESSED_DIR / "metadata.csv"
    
    # Áudio
    TARGET_SR = 16000
    RESAMPLE_TYPE = 'kaiser_best'
    
    # Data Augmentation
    FONEMAS_RAROS = ['ɟ', 'x', 'ʁ', 'ʝ', 'c', 'ç', 'ʒ', 'ɬ']
    AUGMENT_FACTOR = 3
    
    # Aumentos
    PITCH_SHIFTS = [-2, -1, 1, 2]
    SPEED_FACTORS = [0.9, 1.1]
    NOISE_SNR = 20
    
    # Idade 8-11
    IDADE_RARA = '8-11'
    OVERSAMPLE_FACTOR = 3
    
    @classmethod
    def create_dirs(cls):
        """Cria todos os diretórios necessários"""
        cls.PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
        cls.PROCESSED_AUDIO_DIR.mkdir(exist_ok=True, parents=True)
        print(f"📁 Diretórios criados/verificados:")
        print(f"   • {cls.PROCESSED_DIR}")
        print(f"   • {cls.PROCESSED_AUDIO_DIR}")

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
    def resample_and_save(audio_id, audio_path, output_dir):
        """
        Resample de alta qualidade e salva como WAV
        Retorna: (caminho_do_arquivo, success_bool)
        """
        output_path = output_dir / f"{audio_id}.wav"
        
        # Se já existe, retorna sucesso
        if output_path.exists():
            return str(output_path), True
        
        try:
            # Carregar com taxa original
            y, orig_sr = librosa.load(audio_path, sr=None)
            
            # Resample se necessário
            if orig_sr != Config.TARGET_SR:
                y = librosa.resample(y, orig_sr=orig_sr, target_sr=Config.TARGET_SR, 
                                    res_type=Config.RESAMPLE_TYPE)
            
            # Salvar
            sf.write(output_path, y, Config.TARGET_SR)
            return str(output_path), True
            
        except Exception as e:
            print(f"   ⚠️ Erro em {audio_id}: {str(e)[:50]}...")
            return None, False

# ============================================
# 3. DATA AUGMENTATION
# ============================================
class AudioAugmenter:
    @staticmethod
    def add_noise(y, sr, snr_db=Config.NOISE_SNR):
        signal_power = np.mean(y**2)
        noise_power = signal_power / (10**(snr_db/10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(y))
        return y + noise
    
    @staticmethod
    def pitch_shift(y, sr, n_steps):
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    
    @staticmethod
    def time_stretch(y, speed_factor):
        return librosa.effects.time_stretch(y, rate=speed_factor)
    
    @staticmethod
    def generate_augmented_samples(y, sr, augment_type='all'):
        augmented = []
        
        if augment_type in ['all', 'pitch']:
            for shift in Config.PITCH_SHIFTS:
                try:
                    y_pitch = AudioAugmenter.pitch_shift(y, sr, shift)
                    augmented.append((y_pitch, f"pitch_{shift}"))
                except:
                    pass
        
        if augment_type in ['all', 'speed']:
            for factor in Config.SPEED_FACTORS:
                try:
                    y_speed = AudioAugmenter.time_stretch(y, factor)
                    augmented.append((y_speed, f"speed_{factor}"))
                except:
                    pass
        
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
        # Verificar dependências
        if not check_dependencies():
            raise ImportError("Pacotes necessários não instalados")
        
        # Criar diretórios
        Config.create_dirs()
        
        # Carregar dados
        print("\n📂 Carregando transcrições...")
        self.df = DataLoader.load_transcriptions()
        print(f"   ✅ Total: {len(self.df)} amostras")
        
        # Identificar amostras especiais
        print("\n🔍 Identificando amostras para augmentation...")
        self.fonemas_raros_ids = DataLoader.analyze_fonemas_raros(self.df)
        
        self.idade_rara_ids = self.df[self.df['age_bucket'] == Config.IDADE_RARA]['utterance_id'].tolist()
        print(f"   • {len(self.idade_rara_ids)} amostras idade {Config.IDADE_RARA}")
        
        todos_fonemas_raros = set()
        for ids in self.fonemas_raros_ids.values():
            todos_fonemas_raros.update(ids)
        print(f"   • {len(todos_fonemas_raros)} amostras com fonemas raros")
        
        self.todos_fonemas_raros = todos_fonemas_raros
    
    def process_audio_file(self, audio_id):
        """Processa um único arquivo de áudio (versão base)"""
        audio_path = Config.RAW_AUDIO_DIR / f"{audio_id}.flac"
        
        if not audio_path.exists():
            print(f"   ⚠️ Áudio não encontrado: {audio_id}")
            return None
        
        # Processar e salvar
        path, success = AudioProcessor.resample_and_save(
            audio_id, audio_path, Config.PROCESSED_AUDIO_DIR
        )
        
        if success:
            return {
                'id': audio_id,
                'path': path,
                'augmented': False
            }
        return None
    
    def generate_augmented_versions(self, audio_id, augment_type='all'):
        """Gera versões aumentadas de um áudio"""
        base_path = Config.PROCESSED_AUDIO_DIR / f"{audio_id}.wav"
        
        if not base_path.exists():
            return []
        
        try:
            y, _ = librosa.load(base_path, sr=Config.TARGET_SR)
        except Exception as e:
            print(f"   ⚠️ Erro ao carregar {audio_id} para augmentation: {e}")
            return []
        
        # Gerar versões aumentadas
        augmented = AudioAugmenter.generate_augmented_samples(y, Config.TARGET_SR, augment_type)
        
        results = []
        for y_aug, aug_name in augmented:
            aug_path = Config.PROCESSED_AUDIO_DIR / f"{audio_id}_{aug_name}.wav"
            try:
                sf.write(aug_path, y_aug, Config.TARGET_SR)
                results.append({
                    'id': f"{audio_id}_{aug_name}",
                    'original_id': audio_id,
                    'path': str(aug_path),
                    'augmented': True,
                    'augmentation': aug_name
                })
            except Exception as e:
                print(f"   ⚠️ Erro ao salvar {aug_name}: {e}")
        
        return results
    
    def run(self):
        """Executa pipeline completo"""
        print("\n" + "="*60)
        print("🚀 INICIANDO PIPELINE DE PRÉ-PROCESSAMENTO")
        print("="*60)
        
        # Processar todas as amostras base
        print("\n🔄 Processando áudios base...")
        todas_amostras = []
        
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processando"):
            audio_id = row['utterance_id']
            
            # Versão base
            base = self.process_audio_file(audio_id)
            if base:
                todas_amostras.append(base)
        
        print(f"\n✅ {len(todas_amostras)} áudios base processados com sucesso")
        
        # Gerar augmentations
        print("\n🔄 Gerando versões aumentadas...")
        
        # Para fonemas raros (augmentation completa)
        for audio_id in tqdm(self.todos_fonemas_raros, desc="Fonemas raros"):
            for _ in range(Config.AUGMENT_FACTOR):
                augmented = self.generate_augmented_versions(audio_id, 'all')
                todas_amostras.extend(augmented)
        
        # Para idade rara (augmentation suave)
        for audio_id in tqdm(self.idade_rara_ids, desc="Idade rara"):
            if audio_id not in self.todos_fonemas_raros:
                for _ in range(Config.OVERSAMPLE_FACTOR):
                    augmented = self.generate_augmented_versions(audio_id, 'pitch')
                    todas_amostras.extend(augmented)
        
        # Criar DataFrame final
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
        df_metadata.to_csv(Config.METADATA_FILE, index=False)
        
        # Estatísticas finais
        print("\n" + "="*60)
        print("✅ PIPELINE CONCLUÍDO!")
        print("="*60)
        print(f"""
📈 ESTATÍSTICAS FINAIS:
   • Amostras originais: {len(self.df)}
   • Amostras após pipeline: {len(df_metadata)}
   • Fator de aumento: {len(df_metadata)/len(self.df):.1f}x
   
   • Amostras aumentadas: {df_metadata['augmented'].sum()}
   • Amostras não aumentadas: {(~df_metadata['augmented']).sum()}
   
📁 Arquivos salvos:
   • Áudios: {Config.PROCESSED_AUDIO_DIR}/
   • Metadados: {Config.METADATA_FILE}
        """)
        
        return df_metadata

# ============================================
# 5. FUNÇÕES DE UTILIDADE
# ============================================
class DatasetUtils:
    @staticmethod
    def load_metadata():
        if not Config.METADATA_FILE.exists():
            raise FileNotFoundError("Execute o pipeline primeiro!")
        return pd.read_csv(Config.METADATA_FILE)
    
    @staticmethod
    def create_train_val_split(df, val_split=0.1, stratify_by='age_bucket'):
        from sklearn.model_selection import train_test_split
        
        unique_ids = df['original_id'].unique()
        
        stratify = None
        if stratify_by:
            idade_por_id = df.groupby('original_id')['age_bucket'].first()
            stratify = idade_por_id.loc[unique_ids].values
        
        ids_train, ids_val = train_test_split(
            unique_ids, test_size=val_split, random_state=42, stratify=stratify
        )
        
        train_df = df[df['original_id'].isin(ids_train)]
        val_df = df[df['original_id'].isin(ids_val)]
        
        print(f"\n📊 Split treino/validação:")
        print(f"  • Treino: {len(train_df)} amostras ({len(ids_train)} originais)")
        print(f"  • Validação: {len(val_df)} amostras ({len(ids_val)} originais)")
        
        return train_df, val_df
    
    @staticmethod
    def get_audio_and_transcript(row):
        audio_path = row['audio_path']
        y, sr = librosa.load(audio_path, sr=Config.TARGET_SR)
        text = row['phonetic_text']
        return y, text

# ============================================
# EXECUÇÃO
# ============================================
if __name__ == "__main__":
    pipeline = PreprocessingPipeline()
    df_metadata = pipeline.run()