import json
import sys

if __name__ == "__main__":
    gsm8k_list = json.load(open(sys.argv[1], "r"))
    longest_list = json.load(open(sys.argv[2], "r"))
    final_list = longest_list + gsm8k_list

    json.dump(final_list,
              open(sys.argv[3], "w"))
