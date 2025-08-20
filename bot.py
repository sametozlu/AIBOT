import os
import io
import asyncio
from typing import List, Tuple

import numpy as np
from PIL import Image

# Try to import TensorFlow for model loading
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow yüklendi")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("❌ TensorFlow yüklenemedi")

import discord
from discord import app_commands


MODEL_FILENAME = "keras_model.h5"
LABELS_FILENAME = "labels.txt"


def load_labels(labels_path: str) -> List[str]:
    try:
        labels: List[str] = []
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                # Many export tools prefix lines with an index (e.g., "0 cat")
                parts = text.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    text = " ".join(parts[1:]).strip()
                labels.append(text)
        if not labels:
            raise ValueError("Labels file is empty.")
        print(f"📝 Etiketler yüklendi: {labels}")
        return labels
    except Exception as e:
        print(f"Labels yüklenemedi: {e}")
        # Fallback labels - kedi odaklı
        fallback_labels = ["Yavru Kedi", "Yavru Kedi Yapay Zeka", "Kedi", "Köpek", "Diğer"]
        print(f"🔄 Fallback etiketler kullanılıyor: {fallback_labels}")
        return fallback_labels


class ImageClassifier:
    def __init__(self, model_path: str, labels_path: str):
        self.model_path = model_path
        self.labels = load_labels(labels_path)
        
        # Try to load the actual model
        if TENSORFLOW_AVAILABLE:
            try:
                print("🤖 Keras modeli yükleniyor...")
                
                # Suppress TensorFlow warnings
                tf.get_logger().setLevel('ERROR')
                
                # Load the model with custom objects to handle compatibility issues
                def custom_depthwise_conv2d(*args, **kwargs):
                    # Remove problematic 'groups' parameter for older models
                    if 'groups' in kwargs:
                        del kwargs['groups']
                    from tensorflow.keras.layers import DepthwiseConv2D
                    return DepthwiseConv2D(*args, **kwargs)
                
                custom_objects = {
                    'DepthwiseConv2D': custom_depthwise_conv2d
                }
                
                self.model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
                
                # Get model input shape
                input_shape = self.model.inputs[0].shape
                self.input_height = int(input_shape[1]) if input_shape[1] is not None else 224
                self.input_width = int(input_shape[2]) if input_shape[2] is not None else 224
                self.input_channels = int(input_shape[3]) if input_shape[3] is not None else 3
                
                print(f"✅ Model başarıyla yüklendi!")
                print(f"📐 Input boyutu: {self.input_width}x{self.input_height}x{self.input_channels}")
                print(f"🏷️ Etiketler: {self.labels}")
                
            except Exception as e:
                print(f"❌ Model yüklenemedi: {e}")
                print("🔍 Hata detayı: Model yapısında uyumsuzluk var")
                print("💡 Çözüm: Model dosyası eski TensorFlow versiyonuyla kaydedilmiş")
                print("🔄 Test modunda çalışıyor...")
                
                # Fallback to test mode
                self.model = None
                self.input_height = 224
                self.input_width = 224
                self.input_channels = 3
        else:
            print("🔄 TensorFlow yok, test modunda çalışıyor...")
            self.model = None
            self.input_height = 224
            self.input_width = 224
            self.input_channels = 3

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        # Convert to RGB and resize to model's expected input size
        image_rgb = image.convert("RGB").resize(
            (self.input_width, self.input_height), Image.Resampling.LANCZOS
        )
        image_array = np.asarray(image_rgb).astype(np.float32)

        # Teachable Machine style normalization: [-1, 1]
        normalized = (image_array / 127.5) - 1.0

        # Add batch dimension
        data = np.expand_dims(normalized, axis=0)
        return data

    def predict(self, image_bytes: bytes, top_k: int = 3) -> List[Tuple[str, float]]:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            input_tensor = self._preprocess(image)
            
            if self.model is not None:
                # Use real model
                predictions = self.model.predict(input_tensor, verbose=0)
                if predictions.ndim == 2:
                    predictions = predictions[0]
            else:
                # Use smart dummy predictions for test mode
                # Give higher probability to cat-related labels
                predictions = np.random.random(len(self.labels))
                
                # Boost cat-related labels
                for i, label in enumerate(self.labels):
                    if "kedi" in label.lower() or "cat" in label.lower():
                        predictions[i] *= 3.0  # Boost cat labels
                    elif "köpek" in label.lower() or "dog" in label.lower():
                        predictions[i] *= 1.5  # Slight boost to dog labels
                
                predictions = predictions / np.sum(predictions)  # Normalize
            
            # Get top k predictions
            num_classes = min(len(self.labels), predictions.shape[0])
            indices = np.argsort(predictions[:num_classes])[::-1][:top_k]
            
            results: List[Tuple[str, float]] = []
            for idx in indices:
                label = self.labels[idx]
                confidence = float(predictions[idx])
                results.append((label, confidence))
            
            return results
            
        except Exception as e:
            print(f"Prediction error: {e}")
            # Return fallback result
            return [("hata", 1.0)]


class ImageClassifierBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.classifier = ImageClassifier(
            model_path=os.path.join(base_dir, MODEL_FILENAME),
            labels_path=os.path.join(base_dir, LABELS_FILENAME),
        )

        @self.tree.command(name="tani", description="Resim yükle ve modelin tahminini al")
        @app_commands.describe(resim="Sınıflandırılacak resim dosyası")
        async def tani(interaction: discord.Interaction, resim: discord.Attachment):
            await interaction.response.defer(thinking=True)

            if not resim.content_type or not resim.content_type.startswith("image/"):
                await interaction.followup.send("❌ Lütfen bir resim dosyası yükleyin.")
                return

            try:
                image_bytes = await resim.read()

                # Run blocking prediction in a worker thread
                results = await asyncio.to_thread(self.classifier.predict, image_bytes, 3)

                top_label, top_conf = results[0]
                details = "\n".join(
                    [f"• {label}: {conf * 100:.2f}%" for label, conf in results]
                )
                
                if self.classifier.model is not None:
                    message = f"🔍 **Resim Analizi Sonucu**\n\n"
                    message += f"📊 **En olası sınıf:** {top_label} ({top_conf * 100:.2f}%)\n\n"
                    message += f"📋 **Tüm tahminler:**\n{details}"
                else:
                    message = f"🔍 **Test Modu - Resim Analizi**\n\n"
                    message += f"📊 **Simüle edilen sonuç:** {top_label} ({top_conf * 100:.2f}%)\n\n"
                    message += f"📋 **Simüle edilen tahminler:**\n{details}\n\n"
                    message += f"⚠️ **Not:** Bu test modunda çalışıyor. Gerçek model henüz yüklenmedi."
                
                await interaction.followup.send(message)
            except Exception as exc:
                await interaction.followup.send(
                    f"❌ Bir hata oluştu: {type(exc).__name__}: {exc}"
                )

    async def setup_hook(self) -> None:
        # Sync application commands (slash commands)
        guild_id_env = os.getenv("DISCORD_GUILD_ID")
        if guild_id_env:
            try:
                guild_id = int(guild_id_env)
                guild = discord.Object(id=guild_id)
                await self.tree.sync(guild=guild)
                print(f"✅ Commands synced to guild: {guild_id}")
            except ValueError:
                await self.tree.sync()
                print("✅ Commands synced globally")
        else:
            await self.tree.sync()
            print("✅ Commands synced globally")


def main() -> None:
    token = "MTM4NjM4Mjc3MzIwMzUwNTIyMg.G_4SLp.sBxhDtSr2xU8886aUWe9sUd8pvAp53umOlcI7k"
    
    print("🤖 Discord Bot başlatılıyor...")
    print("📁 Model dosyası:", MODEL_FILENAME)
    print("📝 Etiket dosyası:", LABELS_FILENAME)
    
    client = ImageClassifierBot()
    client.run(token)


if __name__ == "__main__":
    main()