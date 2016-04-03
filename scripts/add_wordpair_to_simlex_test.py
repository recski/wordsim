import argparse
import os.path

# Argument parser initialization
parser = argparse.ArgumentParser();
parser.add_argument("-w1", "--word1",
                    help="First word of wordpair to add.", type=str)
parser.add_argument("-w2", "--word2",
                    help="Second word of wordpair to add.", type=str)
args = parser.parse_args();

header_str = 'word1\tword2\tPOS\tSimLex999\tconc(w1)\tconc(w2)\t' \
             'concQAssoc(USF)\tSimAssoc333\tSD(SimLex)\n'
str_to_write = ''

if not os.path.exists('resources/sim_data/simlex/SimLex_test.txt'):
    with open('resources/sim_data/simlex/SimLex_test.txt', 'w') as f:
        f.write(header_str)

with open('resources/sim_data/simlex/SimLex_test.txt', 'a+') as sim_test_file:
    existing_words = sim_test_file.readlines()

    with open('resources/sim_data/simlex/SimLex-999.txt', 'r') as f:
        content = f.readlines()
        for line in content:
            if args.word1 in line and args.word2 in line:
                already_in = False
                for line2 in existing_words:
                    if args.word1 in line2 and args.word2 in line2:
                        already_in = True
                        break
                if not already_in:
                    str_to_write += line
                break

    sim_test_file.write(str_to_write)