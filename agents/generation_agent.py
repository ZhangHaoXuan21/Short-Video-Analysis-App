from transformers import BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from pathlib import Path


def get_hugface_model() -> ChatHuggingFace:
    # === 1. Define model and directory ===

    # Qwen/Qwen3-4B-Thinking-2507,  Qwen/Qwen3-4B-Instruct-2507 , HuggingFaceTB/SmolLM2-1.7B-Instruct , Qwen/Qwen3-1.7B, HuggingFaceTB/SmolLM3-3B
    model_name = "Qwen/Qwen3-1.7B"
    model_dir = Path(__file__).parent.parent / "model" / "Qwen3-1.7B"
    #save_dir = r"C:\Interview Assignment\Intel GenAI Software Engineer Assessment\Short-Video-Analyst\model\Qwen3-4B-Instruct-2507"

    # === 2. Configure 4-bit quantization ===
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="float16",
    )

    llm = HuggingFacePipeline.from_model_id(
        model_id=model_name,
        task="text-generation",
        pipeline_kwargs=dict(
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
            return_full_text=False,
        ),
        model_kwargs={
            "cache_dir": model_dir,
            "quantization_config": bnb_config,
            "device_map": "auto",
        },
    )

    chat_model = ChatHuggingFace(llm=llm)

    return chat_model
