from utils.anno_read import anno_read
import sys
import os

if __name__ == "__main__":
    cur_path = os.path.abspath(os.path.dirname(__file__))
    af_path = os.path.join(cur_path, "data/anno/Instance_segmentation.json")
    anno = anno_read(annotation_file=af_path)
    # print(anno.dataset)
    print(len(anno.dataset))
    print(len(anno.dataset["images"]))
    print(anno.dataset["images"][1])
    print(len(anno.dataset["segmentation"]))
    print(anno.dataset["segmentation"][1])