"""
Get metadata about annotations and evaluate the annotators agreement.
"""

import os

PATH_TO_ORIGINAL = os.path.join(
    os.path.basename(__file__),
    "data",
    "annotations",
    "original"
)
PATH_TO_ANNOTATED = os.path.join(
    os.path.basename(__file__),
    "data",
    "annotations",
    "annotated"
)

def main():

    assert os.path.exists(PATH_TO_ORIGINAL)
    assert os.path.exists(PATH_TO_ANNOTATED)

    

if __name__ == "__main__":
    main()