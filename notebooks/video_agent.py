from __future__ import annotations
from typing import Literal, Any
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from pathlib import Path


class SmolVLM2ChatModel:
    """
    A LangChain-style wrapper for the HuggingFace SmolVLM2 family of models that supports
    conversational `.invoke()` and `.batch()` calls.

    This class automatically selects a model variant (`256M`, `500M`, or `2.2B`) based on
    GPU VRAM if `model_size` is not specified, and supports 4-bit or 8-bit quantization
    using `bitsandbytes`.

    Attributes:
        device (str): Target device for model execution (e.g., "cuda" or "cpu").
        quantization (str): Quantization type ("none", "8bit", or "4bit").
        model_name (str): Full Hugging Face model ID.
        model_dir (Path): Local directory for caching model files.
        processor (AutoProcessor): Hugging Face processor for text-image inputs.
        model (AutoModelForImageTextToText): Loaded multimodal model instance.
    """

    MODEL_MAP: dict[str, str] = {
        "small": "HuggingFaceTB/SmolVLM2-256M-Instruct",
        "medium": "HuggingFaceTB/SmolVLM2-500M-Instruct",
        "large": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    }

    def __init__(
        self,
        model_size: Literal["small", "medium", "large"] | None = None,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        quantization: Literal["none", "8bit", "4bit"] = "none",
        model_root: str | Path | None = None,
    ) -> None:
        """
        Initialize the SmolVLM2ChatModel.

        Args:
            model_size: The size of the model to load ("small", "medium", or "large").
                If None, the appropriate model will be selected based on available VRAM.
            device: Target device for computation (e.g., "cuda", "cpu"). Defaults to auto-detect.
            dtype: Torch dtype for model weights. Defaults to `torch.bfloat16`.
            quantization: Quantization mode ("none", "8bit", or "4bit"). Defaults to "none".
            model_root: Optional path to a local root folder for storing model checkpoints.
        """
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.quantization: str = quantization

        # === 1. Auto-select model size if not given ===
        if model_size is None:
            model_size = self._auto_select_model_size()

        self.model_name: str = self.MODEL_MAP.get(model_size, self.MODEL_MAP["medium"])
        print(f"🔹 Preparing to load {self.model_name} ({model_size}) with {quantization} quantization...")

        # === 2. Define local cache directory ===
        model_root = Path(model_root) if model_root else Path(__file__).parent.parent / "model"
        self.model_dir: Path = model_root / self.model_name.split("/")[-1]
        self.model_dir.mkdir(parents=True, exist_ok=True)
        print(f"📂 Model cache directory: {self.model_dir}")

        # === 3. Configure quantization ===
        quant_config: BitsAndBytesConfig | None = None
        if quantization in ["8bit", "4bit"]:
            try:
                quant_config = BitsAndBytesConfig(
                    load_in_8bit=(quantization == "8bit"),
                    load_in_4bit=(quantization == "4bit"),
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                print(f"⚙️ Using bitsandbytes {quantization} quantization.")
            except Exception as e:
                print(f"⚠️ Quantization unavailable ({e}), loading without it.")
                quant_config = None

        # === 4. Load processor ===
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir=self.model_dir,
        )

        # === 5. Load model ===
        print(f"🚀 Loading model weights into {self.device}...")
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            cache_dir=self.model_dir,
            torch_dtype=dtype,
            quantization_config=quant_config,
            device_map="auto" if quant_config else None,
        )

        # === 6. Move to device if not quantized ===
        if not quant_config:
            self.model.to(self.device)

        print(f"✅ Successfully loaded {model_size.upper()} SmolVLM2 model ({self.device})!")

    # -------------------------------------------------------------------------
    def _auto_select_model_size(self) -> Literal["small", "medium", "large"]:
        """
        Automatically determine which model size to use based on available GPU VRAM.

        Returns:
            The recommended model size as a string literal.
        """
        if not torch.cuda.is_available():
            print("⚠️ CUDA not available. Using CPU-friendly 256M model.")
            return "small"

        try:
            total_vram_gb: float = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🧮 Detected VRAM: {total_vram_gb:.1f} GB")

            if total_vram_gb < 6:
                return "small"
            elif total_vram_gb < 10:
                return "medium"
            else:
                return "large"
        except Exception as e:
            print("⚠️ Could not detect VRAM:", e)
            return "medium"

    # -------------------------------------------------------------------------
    def _format_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Validate and prepare chat messages for model input.

        Args:
            messages: List of chat message dictionaries.

        Returns:
            The validated list of messages.
        """
        if not isinstance(messages, list):
            raise ValueError("messages must be a list of message dicts.")
        return messages

    # -------------------------------------------------------------------------
    def _generate(
        self,
        formatted_messages: list[dict[str, Any]],
        max_new_tokens: int = 512,
        do_sample: bool = False,
    ) -> str:
        """
        Run text generation on formatted chat messages.

        Args:
            formatted_messages: A list of formatted chat messages.
            max_new_tokens: Maximum number of tokens to generate. Defaults to 512.
            do_sample: Whether to enable sampling (for creativity). Defaults to False.

        Returns:
            The decoded model response as a string.
        """
        inputs = self.processor.apply_chat_template(
            formatted_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device, dtype=torch.bfloat16)

        outputs = self.model.generate(
            **inputs,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
        )

        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

    # -------------------------------------------------------------------------
    def invoke(self, input: dict[str, Any]) -> dict[str, str]:
        """
        Perform a single conversational inference turn.

        Args:
            input: A single message dictionary (e.g., {"role": "user", "content": "Describe the image"}).

        Returns:
            A dictionary representing the AI model's reply.
        """
        formatted_messages = self._format_messages([input])
        return {"type": "ai_message", "content": self._generate(formatted_messages)}

    # -------------------------------------------------------------------------
    def batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, str]]:
        """
        Perform multiple chat turns in a batch.

        Args:
            inputs: List of message dictionaries.

        Returns:
            A list of dictionaries containing AI responses.
        """
        results: list[dict[str, str]] = []
        for inp in inputs:
            formatted_messages = self._format_messages([inp])
            response = self._generate(formatted_messages)
            results.append({"type": "ai_message", "content": response})
        return results
