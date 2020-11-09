from os import environ, path, mkdir


VOS_KIND_TO_TEST = environ.get("VOS_INPUT_KIND", "PREMVOS")
OUTPUT_METRICS_PATH = path.join(
    environ.get("OUTPUT_METRICS_PATH"),
    VOS_KIND_TO_TEST)
if not path.exists(OUTPUT_METRICS_PATH):
    mkdir(OUTPUT_METRICS_PATH)
