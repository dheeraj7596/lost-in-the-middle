import json

if __name__ == "__main__":
    f = open("/Users/dheerajmekala/Work/lost-in-the-middle/training/data/gsm8k_1000_distractors.jsonl", "r")

    longest_list = json.load(open("/Users/dheerajmekala/Work/lost-in-the-middle/training/data/longest_5667.json", "r"))
    lines = f.readlines()
    for l in lines:
        temp = json.loads(l.strip())
        longest_list.append(temp)

    json.dump(longest_list,
              open("/Users/dheerajmekala/Work/lost-in-the-middle/training/data/longest_distractods.json", "w"))
