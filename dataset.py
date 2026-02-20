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
    """Loads attacker prompt templates for RL red-teaming."""

    def __init__(self, jsonl_file: str):
        self.prompts = []
        with open(jsonl_file, "r") as f:
            for line in f:
                self.prompts.append(json.loads(line)["attacker_prompt"])

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        return self.prompts[index]

    def get_batch(self, batch_size: int) -> List[str]:
        return [random.choice(self.prompts) for _ in range(batch_size)]
