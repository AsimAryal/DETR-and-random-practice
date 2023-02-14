from pycocotools.coco import COCO


def get_coco_objects(path):
    return COCO(path)


def get_coco_image_annotation(coco_obj):
    image_ids = coco_obj.getImgIds()
    images = coco_obj.loadImgs(image_ids)
    list_of_annotations = []
    for image_id in image_ids:
        annotation_ids = coco_obj.getAnnIds(imgIds=image_id, iscrowd=None)
        annotations = coco_obj.loadAnns(annotation_ids)
        for ann in annotations:
            list_of_annotations.append(
                {
                    "image_id": image_id,
                    "bbox": ann["bbox"],
                    "category_id": ann["category_id"],
                }
            )
    return images, list_of_annotations
