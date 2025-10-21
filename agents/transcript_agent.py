from __future__ import annotations
from typing import Optional
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from moviepy import VideoFileClip
from pathlib import Path
import os


class VoiceToText:
    """
    A utility class for automatic speech recognition (ASR) using Whisper models.

    This class can:
      • Extract audio from video files (e.g., MP4 → WAV)
      • Transcribe audio to text using Hugging Face Whisper models
      • Automatically clean up temporary files

    Attributes:
        processor (WhisperProcessor): The Hugging Face processor for Whisper.
        model (WhisperForConditionalGeneration): The loaded Whisper model instance.
        device (str): The target device for inference ("cuda" or "cpu").
        model_dir (Path): Local cache directory for model files.
    """

    # openai/whisper-tiny openai/whisper-tiny.en
    def __init__(
        self,
        model_name: str = "openai/whisper-tiny",
        device: Optional[str] = None,
        model_root: Optional[str | Path] = None,
    ) -> None:
        """
        Initialize the Whisper voice-to-text transcriber.

        Args:
            model_name: The Hugging Face model ID to load (e.g., "openai/whisper-tiny.en").
            device: Target device for computation. Defaults to "cuda" if available.
            model_root: Optional local directory for caching model weights.
        """
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # === Define local cache directory ===
        model_root = Path(model_root) if model_root else Path(__file__).parent.parent / "model"
        self.model_dir: Path = model_root / model_name.replace("/", "-")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        print(f"🔹 Loading Whisper model: {model_name}")
        print(f"📂 Model cache directory: {self.model_dir}")

        # === Load model and processor ===
        self.processor: WhisperProcessor = WhisperProcessor.from_pretrained(
            model_name,
            cache_dir=self.model_dir,
        )
        self.model: WhisperForConditionalGeneration = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=self.model_dir,
        ).to(self.device)

        print(f"✅ Whisper model loaded successfully on {self.device}")

    # -------------------------------------------------------------------------
    def extract_audio_from_video(self, video_path: str | Path) -> str:
        """
        Extract the audio track from a video file and save it as a WAV file.

        The output audio file is saved in the same directory as the video.

        Args:
            video_path: Path to the input video file (e.g., ".mp4").

        Returns:
            The path to the extracted WAV file as a string.

        Raises:
            ValueError: If the video contains no audio stream.
        """
        video_path = Path(video_path)
        output_wav_path = video_path.with_suffix(".wav")

        clip = VideoFileClip(str(video_path))

        try:
            # Handle videos with no audio
            if clip.audio is None:
                raise ValueError(f"No audio stream found in video: {video_path.name}")

            clip.audio.write_audiofile(str(output_wav_path), codec="pcm_s16le")
            print(f"🎵 Extracted audio to: {output_wav_path}")

        finally:
            clip.close()

        return str(output_wav_path)

    # -------------------------------------------------------------------------
    def transcribe(
        self,
        file_path: str | Path,
        chunk_length_s: int = 30,
    ) -> str:
        """
        Transcribe an audio or video file to text.

        Automatically handles:
          • MP4 → WAV audio extraction
          • Chunked transcription for long audio
          • Temporary WAV cleanup

        Args:
            file_path: Path to the audio or video file.
            chunk_length_s: Length (in seconds) of each chunk for transcription. Defaults to 30s.

        Returns:
            The full transcribed text as a string.

        Raises:
            ValueError: If no audio data is found in the file.
        """
        file_path = Path(file_path)
        temp_audio_path: Optional[str] = None

        try:
            # === 1. If video, extract temporary audio file ===
            if file_path.suffix.lower() == ".mp4":
                temp_audio_path = self.extract_audio_from_video(file_path)
                file_path = Path(temp_audio_path)

            # === 2. Load audio ===
            speech_array, sampling_rate = torchaudio.load(str(file_path))
            if speech_array.numel() == 0:
                raise ValueError(f"The file '{file_path}' contains no audio data.")

            # === 3. Resample if needed ===
            if sampling_rate != 16000:
                resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
                speech_array = resampler(speech_array)
                sampling_rate = 16000

            # === 4. Process chunks ===
            chunk_size = chunk_length_s * sampling_rate
            all_text: list[str] = []

            for i in range(0, speech_array.shape[1], chunk_size):
                chunk = speech_array[:, i:i + chunk_size]
                if chunk.shape[1] == 0:
                    continue

                input_features = self.processor(
                    chunk.squeeze().numpy(),
                    sampling_rate=sampling_rate,
                    return_tensors="pt",
                ).input_features.to(self.device)

                predicted_ids = self.model.generate(input_features)
                transcription = self.processor.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )[0]
                all_text.append(transcription.strip())

            return " ".join(all_text)

        finally:
            # === 5. Clean up temporary file ===
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                print(f"🧹 Removed temporary file: {temp_audio_path}")
