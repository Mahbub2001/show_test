"""Runs Show-o2 on every GapEval item in both directions:

  - understanding (und): question image + und_prompt -> text answer   (mmu_generate)
  - generation    (gen): gen_prompt -> generated image                (t2i_generate + flow ODE sampler)

Mirrors the official ../inference_mmu.py and ../inference_t2i.py call patterns as closely as
possible so the model is exercised exactly the way the show-o2 authors intended.

Caveat: the publicly released Show-o2 checkpoints only expose *unconditional* text-to-image
generation via t2i_generate (see show-o2 README TODO: "Release the models supporting
mixed-modality generation" is still unchecked). So for GapEval items whose gen_prompt asks to
edit/transform the *given* question image (Instruction Following, some Reasoning/Physics items),
this script generates from gen_prompt text alone -- the question image is not fed into the
generation path. This matches what the base t2i_generate interface supports today; it is not a
bug in this script.

Usage (run from the show-o2/ directory, or pass --show_o2_root):
    python gapeval_eval/showo2_runner.py \
        --config configs/showo2_1.5b_demo_432x432.yaml \
        --data_dir gapeval_eval/data \
        --output_dir gapeval_eval/outputs/showo2_1.5b \
        --direction both
"""
import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHOWO2_ROOT = os.path.dirname(SCRIPT_DIR)  # .../show-o2

from dotenv import load_dotenv
load_dotenv()  # picks up show-o2/.env (searches this dir and parents); may set HF_HOME
# Keep all HF downloads (Qwen2.5, SigLIP, show-o2 checkpoints) inside show-o2/ instead of the
# user-global ~/.cache/huggingface -- unless the user already set HF_HOME themselves (.env or shell).
os.environ.setdefault("HF_HOME", os.path.join(SHOWO2_ROOT, "hf_cache"))

import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, SCRIPT_DIR)
from data import GapEvalItem, load_gapeval  # noqa: E402


def _import_showo2():
    """Import show-o2's package tree. Must happen after inserting SHOWO2_ROOT onto sys.path,
    and this local `datasets` package shadows the pip `datasets` package for the duration --
    that's why data.py (which needs the real HF `datasets`) does its own loading with no
    dependency on that package, and why this function is called lazily, not at module import
    time."""
    sys.path.insert(0, SHOWO2_ROOT)
    from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive  # noqa: E401
    from models.misc import get_text_tokenizer, prepare_gen_input  # noqa: E401
    from datasets.utils import image_transform  # noqa: E401
    from utils import get_hyper_params, path_to_llm_name, denorm, load_state_dict  # noqa: E401
    from transport import Sampler, create_transport  # noqa: E401

    return dict(
        Showo2Qwen2_5=Showo2Qwen2_5,
        WanVAE=WanVAE,
        omni_attn_mask_naive=omni_attn_mask_naive,
        get_text_tokenizer=get_text_tokenizer,
        prepare_gen_input=prepare_gen_input,
        image_transform=image_transform,
        get_hyper_params=get_hyper_params,
        path_to_llm_name=path_to_llm_name,
        denorm=denorm,
        load_state_dict=load_state_dict,
        Sampler=Sampler,
        create_transport=create_transport,
    )


class Showo2Eval:
    def __init__(self, config_path: str, device: str = "cuda", weight_type: str = "bfloat16"):
        self.mods = _import_showo2()
        self.config = OmegaConf.load(config_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.weight_type = {"bfloat16": torch.bfloat16, "float32": torch.float32}[weight_type]

        vae_cfg = self.config.model.vae_model
        if vae_cfg.type != "wan21":
            raise NotImplementedError(vae_cfg.type)
        self.vae_model = self.mods["WanVAE"](
            vae_pth=vae_cfg.pretrained_model_path, dtype=self.weight_type, device=self.device
        )

        self.text_tokenizer, self.showo_token_ids = self.mods["get_text_tokenizer"](
            self.config.model.showo.llm_model_path,
            add_showo_tokens=True,
            return_showo_token_ids=True,
            llm_name=self.mods["path_to_llm_name"][self.config.model.showo.llm_model_path],
        )
        self.config.model.showo.llm_vocab_size = len(self.text_tokenizer)

        showo_cfg = self.config.model.showo
        if showo_cfg.load_from_showo:
            self.model = self.mods["Showo2Qwen2_5"].from_pretrained(
                showo_cfg.pretrained_model_path, use_safetensors=False
            ).to(self.device)
        else:
            self.model = self.mods["Showo2Qwen2_5"](**showo_cfg).to(self.device)
            state_dict = self.mods["load_state_dict"](self.config.model_path)
            self.model.load_state_dict(state_dict)
        self.model.to(self.weight_type)
        self.model.eval()

        if self.config.model.showo.add_time_embeds:
            self.config.dataset.preprocessing.num_t2i_image_tokens += 1
            self.config.dataset.preprocessing.num_mmu_image_tokens += 1
            self.config.dataset.preprocessing.num_video_tokens += 1

        (
            self.num_t2i_image_tokens, self.num_mmu_image_tokens, _num_video_tokens,
            self.max_seq_len, self.max_text_len, self.image_latent_dim, self.patch_size,
            self.latent_width, self.latent_height, self.pad_id, self.bos_id, self.eos_id,
            self.boi_id, self.eoi_id, _bov_id, _eov_id, self.img_pad_id, _vid_pad_id,
            self.guidance_scale,
        ) = self.mods["get_hyper_params"](self.config, self.text_tokenizer, self.showo_token_ids)

        self.sys_prompt_ids = self.text_tokenizer(
            "system\nYou are a helpful assistant.<|im_end|>", add_special_tokens=False
        )["input_ids"]
        self.role_a = self.text_tokenizer("\n<|im_start|>user\n", add_special_tokens=False)["input_ids"]
        self.role_b = self.text_tokenizer("\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]

        transport_cfg = self.config.transport
        self.transport = self.mods["create_transport"](
            path_type=transport_cfg.path_type,
            prediction=transport_cfg.prediction,
            loss_weight=transport_cfg.loss_weight,
            train_eps=transport_cfg.train_eps,
            sample_eps=transport_cfg.sample_eps,
            snr_type=transport_cfg.snr_type,
            do_shift=transport_cfg.do_shift,
            seq_len=self.num_t2i_image_tokens,
        )
        self.sampler = self.mods["Sampler"](self.transport)

    # ---------------------------------------------------------------- und --
    @torch.no_grad()
    def understand(self, image_path: str, question: str, max_new_tokens: int = 300, top_k: int = 1) -> str:
        image_ori = Image.open(image_path).convert("RGB")
        image = self.mods["image_transform"](image_ori, resolution=self.config.dataset.preprocessing.resolution)
        image = image.unsqueeze(0).to(self.device)

        image_latents = self.vae_model.sample(image.unsqueeze(2)).squeeze(2).to(self.weight_type)
        image_embeds_und = self.model.image_embedder_und(image_latents)
        image_embeds_gen = self.model.image_embedder_gen(image_latents)
        image_embeds_und = image_embeds_und + self.model.position_embedding(self.model.image_position_ids)
        image_embeds_und = self.model.und_trans(image_embeds_und)["last_hidden_state"]
        image_embeds = self.model.fusion_proj(torch.cat([image_embeds_und, image_embeds_gen], dim=-1))

        input_ids = self.text_tokenizer(question, add_special_tokens=False).input_ids
        text_tokens_a = torch.tensor([self.showo_token_ids["bos_id"]] + self.sys_prompt_ids + self.role_a
                                     ).to(self.device)[None, :]
        text_tokens_b = torch.tensor(
            [self.showo_token_ids["boi_id"], self.showo_token_ids["eoi_id"]] + input_ids + self.role_b
        ).to(self.device)[None, :]
        text_embeds_a = self.model.showo.model.embed_tokens(text_tokens_a)
        text_embeds_b = self.model.showo.model.embed_tokens(text_tokens_b)

        if self.config.model.showo.add_time_embeds:
            time_embeds = self.model.time_embed(torch.Tensor([[1.0]]).to(self.device), text_embeds_a.dtype)
            if hasattr(self.model, "time_embed_proj"):
                time_embeds = self.model.time_embed_proj(time_embeds)
            input_embeds = torch.cat(
                [text_embeds_a, text_embeds_b[:, :1], time_embeds, image_embeds, text_embeds_b[:, 1:]], dim=1
            ).to(self.weight_type)
            modality_positions = torch.tensor(
                [text_tokens_a.shape[1] + 2, self.num_mmu_image_tokens]
            )[None, None, :].to(self.device)
        else:
            input_embeds = torch.cat(
                [text_embeds_a, text_embeds_b[:, :1], image_embeds, text_embeds_b[:, 1:]], dim=1
            ).to(self.weight_type)
            modality_positions = torch.tensor(
                [text_tokens_a.shape[1] + 1, self.num_mmu_image_tokens]
            )[None, None, :].to(self.device)

        attention_mask = self.mods["omni_attn_mask_naive"](
            B=input_embeds.size(0), LEN=input_embeds.size(1), modalities=modality_positions,
            device=self.device, inverted=True
        ).to(input_embeds.dtype)

        output_tokens = self.model.mmu_generate(
            input_embeds=input_embeds, attention_mask=attention_mask, top_k=top_k,
            max_new_tokens=max_new_tokens, eos_token=self.text_tokenizer.eos_token_id,
        )
        output_tokens = torch.stack(output_tokens).squeeze()[None]
        text = self.text_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
        return text

    # ---------------------------------------------------------------- gen --
    @torch.no_grad()
    def generate(self, prompt: str) -> Image.Image:
        cfg = self.config
        batch_text_tokens, batch_text_tokens_null, batch_modality_positions, batch_modality_positions_null = (
            self.mods["prepare_gen_input"](
                [prompt], self.text_tokenizer, self.num_t2i_image_tokens, self.bos_id, self.eos_id,
                self.boi_id, self.eoi_id, self.pad_id, self.img_pad_id, self.max_text_len, self.device,
            )
        )

        z = torch.randn(
            (1, self.image_latent_dim, self.latent_height * self.patch_size, self.latent_width * self.patch_size)
        ).to(torch.bfloat16).to(self.device)

        guidance_scale = self.guidance_scale
        if guidance_scale > 0:
            z = torch.cat([z, z], dim=0)
            text_tokens = torch.cat([batch_text_tokens, batch_text_tokens_null], dim=0)
            modality_positions = torch.cat([batch_modality_positions, batch_modality_positions_null], dim=0)
        else:
            text_tokens = batch_text_tokens
            modality_positions = batch_modality_positions

        block_mask = self.mods["omni_attn_mask_naive"](
            text_tokens.size(0), self.max_seq_len, modality_positions, self.device
        ).to(self.weight_type)

        model_kwargs = dict(
            text_tokens=text_tokens, attention_mask=block_mask, modality_positions=modality_positions,
            output_hidden_states=True, max_seq_len=self.max_seq_len, guidance_scale=guidance_scale,
        )
        sample_fn = self.sampler.sample_ode(
            sampling_method=cfg.transport.sampling_method, num_steps=cfg.transport.num_inference_steps,
            atol=cfg.transport.atol, rtol=cfg.transport.rtol, reverse=cfg.transport.reverse,
            time_shifting_factor=cfg.transport.time_shifting_factor,
        )
        samples = sample_fn(z, self.model.t2i_generate, **model_kwargs)[-1]
        if guidance_scale > 0:
            samples = torch.chunk(samples, 2)[0]

        samples = samples.unsqueeze(2)
        images = self.vae_model.batch_decode(samples).squeeze(2)
        images = self.mods["denorm"](images)
        return Image.fromarray(images[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a showo2_*.yaml config (relative to show-o2/)")
    parser.add_argument("--data_dir", default=os.path.join(SCRIPT_DIR, "data"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--direction", choices=["und", "gen", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N items (debugging)")
    parser.add_argument("--weight_type", choices=["bfloat16", "float32"], default="bfloat16")
    args = parser.parse_args()

    config_path = args.config if os.path.isabs(args.config) else os.path.join(SHOWO2_ROOT, args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    items = load_gapeval(args.data_dir)
    if args.limit:
        items = items[: args.limit]

    evaluator = Showo2Eval(config_path, weight_type=args.weight_type)

    und_out_path = os.path.join(args.output_dir, "und_outputs.jsonl")
    gen_img_dir = os.path.join(args.output_dir, "gen_images")
    gen_out_path = os.path.join(args.output_dir, "gen_outputs.jsonl")
    os.makedirs(gen_img_dir, exist_ok=True)

    if args.direction in ("und", "both"):
        with open(und_out_path, "w", encoding="utf-8") as f:
            for item in tqdm(items, desc="understanding"):
                if not item.input_image_path:
                    continue
                answer = evaluator.understand(item.input_image_path, item.und_prompt)
                f.write(json.dumps({"id": item.id, "und_prompt": item.und_prompt, "answer": answer},
                                   ensure_ascii=False) + "\n")

    if args.direction in ("gen", "both"):
        with open(gen_out_path, "w", encoding="utf-8") as f:
            for item in tqdm(items, desc="generation"):
                image = evaluator.generate(item.gen_prompt)
                img_path = os.path.join(gen_img_dir, f"{item.id}.png")
                image.save(img_path)
                f.write(json.dumps({"id": item.id, "gen_prompt": item.gen_prompt, "image_path": img_path},
                                   ensure_ascii=False) + "\n")

    print(f"Done. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
