import json
import random
from typing import List


class SFTDataset:
    """Loads attacker SFT data: prompt templates + attack instructions."""

    def __init__(self, prompt_file: str, instruction_file: str, split: str = "train"):
        self.prompts = []
        with open(prompt_file, "r") as f:
            for line in f:
                self.prompts.append(json.loads(line)["attacker_prompt"])

        with open(instruction_file, "r") as f:
            instructions = json.load(f)
        self.instructions = [x["instruction"].strip() for x in instructions]

        random.seed(42)
        random.shuffle(self.instructions)
        num_vals = int(len(self.instructions) * 0.1)

        if split == "train":
            self.instructions = self.instructions[num_vals:]
        elif split == "val":
            self.instructions = self.instructions[:num_vals]

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, index):
        prompt = random.choice(self.prompts)
        instruction = self.instructions[index]
        return {"prompt": prompt, "completion": instruction}

    def get_batch(self, batch_size: int) -> List[dict]:
        indices = random.sample(range(len(self)), min(batch_size, len(self)))
        return [self[i] for i in indices]


class SafetyDataset:
    """Loads safety fine-tuning data: harmful prompts + safe responses."""

    def __init__(self, instruction_file: str):
        with open(instruction_file, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return {
            "instruction": item["instruction"],
            "response": item["response"],
        }

    def get_batch(self, batch_size: int) -> List[dict]:
        indices = random.sample(range(len(self)), min(batch_size, len(self)))
        return [self[i] for i in indices]


class RedTeamDataset:
    """Loads attacker prompt templates for RL red-teaming.

    Supports category-stratified sampling for topic diversity.
    """

    def __init__(self, jsonl_file: str, seed_data_file: str = None,
                 stratify_categories: bool = False):
        self.prompts = []
        with open(jsonl_file, "r") as f:
            for line in f:
                self.prompts.append(json.loads(line)["attacker_prompt"])

        self.stratify = stratify_categories
        self.seeds_by_category = {}  # category -> list of seed instructions

        if seed_data_file and stratify_categories:
            with open(seed_data_file, "r") as f:
                seed_data = json.load(f)
            for item in seed_data:
                cat = item.get("category", "S5")
                if cat not in self.seeds_by_category:
                    self.seeds_by_category[cat] = []
                self.seeds_by_category[cat].append(item["instruction"])
            # Ensure all 8 categories exist
            for cat in ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]:
                if cat not in self.seeds_by_category:
                    self.seeds_by_category[cat] = []

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        return self.prompts[index]

    def get_batch(self, batch_size: int) -> List[dict]:
        """Get a batch of prompts, optionally stratified by category.

        Returns list of dicts with keys: prompt_template, seed_instruction, category.
        """
        if not self.stratify or not self.seeds_by_category:
            return [{"prompt_template": random.choice(self.prompts),
                     "seed_instruction": None, "category": None}
                    for _ in range(batch_size)]

        # Round-robin: distribute slots across categories with seeds
        active_cats = [c for c in sorted(self.seeds_by_category)
                       if self.seeds_by_category[c]]
        results = []
        for i in range(batch_size):
            cat = active_cats[i % len(active_cats)]
            seed_instr = random.choice(self.seeds_by_category[cat])
            results.append({
                "prompt_template": random.choice(self.prompts),
                "seed_instruction": seed_instr,
                "category": cat,
            })
        random.shuffle(results)
        return results
