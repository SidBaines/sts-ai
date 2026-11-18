"""World assignments for the paired learning-rate study."""
from sts_rl.paired_endpoints import compare as compare_endpoints, digest

N = 16_384
PRIMARY = 125_000_000_000
CONFIRM = 126_000_000_000


def compare(a_folder, b_folder, start, a_name, b_name, a_ck, b_ck):
    return compare_endpoints(a_folder, b_folder, start, a_name, b_name, a_ck, b_ck, N)
