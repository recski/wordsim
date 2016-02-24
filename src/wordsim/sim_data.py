import logging
import sys
import traceback

class SimDataFormatError(Exception):
    pass

class SimData():

    @staticmethod
    def create_from_file(file_name, data_type):
        sd_type = type_to_class[data_type]
        sd = sd_type()
        with open(file_name) as f:
            for line_no, line in enumerate(f):
                try:
                    parsed_line = sd_type.parse_line(line, line_no)
                    if parsed_line is None:
                        continue
                    w1, w2, sim = parsed_line
                    sorted_pair = tuple(sorted([w1, w2]))
                    sd.pairs[sorted_pair] = float(sim)
                except:
                    traceback.print_exc()
                    raise SimDataFormatError(
                        'error on line {0}: {1}'.format(line_no+1, line))
        logging.info(
            'read similarity data from {0} ({1} word pairs)'.format(
                file_name, len(sd.pairs)))

        return sd

    def __init__(self):
        self.pairs = {}

    def top_pairs(self, n):
        return sorted(self.pairs.items(), key=lambda x: -x[1])[:n]

class SimLexData(SimData):
    @staticmethod
    def parse_line(line, line_no):
        if line_no == 0:
            return
        fields = line.strip().decode('utf-8').split('\t')
        w1, w2 = fields[:2]
        sim = float(fields[3])
        return w1, w2, sim

class WS353Data(SimData):
    @staticmethod
    def parse_line(line, line_no):
        if line_no == 0:
            return
        fields = line.strip().decode('utf-8').split('\t')
        w1, w2 = fields[:2]
        sim = float(fields[2])
        return w1, w2, sim

class MENData(SimData):
    @staticmethod
    def parse_line(line, line_no):
        if line_no == 0:
            return
        fields = line.strip().decode('utf-8').split()
        w1, w2 = fields[:2]
        sim = float(fields[2])
        return w1, w2, sim

type_to_class = {
    'simlex': SimLexData,
    'men': MENData,
    'ws353': WS353Data}

def test():
    """read similarity data from specified file and output top 5 word pairs"""
    sd = SimData.create_from_file(sys.argv[1], sys.argv[2])
    print sd.top_pairs(5)

if __name__ == "__main__":
    test()
