import re


# --- Extracted Dependencies ---

def parse_int_list(s):
    if isinstance(s, list): return s
    if isinstance(s, int): return [s]
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges
