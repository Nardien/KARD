import os
import json
from termcolor import cprint
from datetime import datetime

class ReasoningWriter():
    def __init__(self, args):
        reasoner_base = args.checkpoint_path.split("/")[2]
        knowledge_base = str(args.knowledge_base)
        dataset = args.dataset

        now = datetime.now()
        formatted_date = now.strftime("%Y%m%d")

        master_path = f"{dataset}-{reasoner_base}-{knowledge_base}"
        if args.knowledge_base is not None:
            master_path += f"-{args.retriever_type}"

        if args.ground_truth_doc:
            master_path += "-GT"

        index = 0
        save_path = os.path.join("reasoning", master_path, formatted_date, f"{args.seed}_{index}")

        while os.path.exists(save_path):
            index += 1
            save_path = os.path.join("reasoning", master_path, formatted_date, f"{args.seed}_{index}")

        cprint(f"Save reasoning to {save_path}", 'red')

        os.makedirs(save_path)

        with open(os.path.join(save_path, "args.json"), 'w') as f:
            json.dump(args.__dict__, f, indent=4)

        self.save_path = save_path
        print("Reasoning writer ready.")

    def write(self, args, data_idx, rationales, choices, gold_label, knowledge):
        dump_path = os.path.join(self.save_path, str(data_idx).zfill(5) + ".json")

        dump_data = {
            "gold_label": gold_label,
            "rationales": rationales,
            "choices": choices,
            "knowledge": knowledge,
        }

        with open(dump_path, 'w') as f:
            json.dump(dump_data, f, indent=4)