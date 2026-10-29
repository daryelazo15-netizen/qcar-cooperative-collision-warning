"""Opt-in other-vehicle target view for HEAL's QCar intermediate loader.

Raw YAML/images are preserved. Fused labels exclude reference ego by ID;
single-observer occupancy supervision retains each observer's own semantics.
The known-body anchor region is ignored, not forced to background, matching
the postprocessing own-body exclusion used by the existing QCar visualization.
"""
import numpy as np
from contextvars import ContextVar
import torch
import opencood.data_utils.datasets as dataset_module
from opencood.utils import box_utils


active_radius = ContextVar("qcar_reference_ego_radius", default=None)
original_nms = box_utils.nms_rotated


def nms_without_reference_ego(boxes, scores, threshold):
    radius = active_radius.get()
    if radius is None:
        return original_nms(boxes, scores, threshold)
    keep = boxes.mean(dim=1)[:, :2].norm(dim=1) >= radius
    indices = torch.nonzero(keep, as_tuple=False).flatten()
    if not len(indices):
        return indices
    return indices[original_nms(boxes[indices], scores[indices], threshold)]


box_utils.nms_rotated = nms_without_reference_ego


def other_vehicle_targets(ego, anchors, generate_label, radius_model):
    reference = str(ego["cav_id_list"][0])
    ids = ego["object_ids"]
    indices = np.flatnonzero(ego["object_bbx_mask"] > 0)
    if len(ids) != len(indices):
        raise ValueError("Object ID/valid-box alignment differs from expected HEAL format")
    keep = [i for i, object_id in enumerate(ids) if str(object_id) != reference]
    boxes = np.zeros_like(ego["object_bbx_center"])
    mask = np.zeros_like(ego["object_bbx_mask"])
    if keep:
        boxes[:len(keep)] = ego["object_bbx_center"][indices[keep]]
        mask[:len(keep)] = 1
    labels = generate_label(gt_box_center=boxes, anchors=anchors, mask=mask)
    ignored = np.linalg.norm(anchors[..., :2], axis=-1) < radius_model
    labels["pos_equal_one"][ignored] = 0
    labels["neg_equal_one"][ignored] = 0
    ego.update(object_bbx_center=boxes, object_bbx_mask=mask,
               object_ids=[ids[i] for i in keep], label_dict=labels,
               reference_ego_id=reference)
    return len(ids) - len(keep)


original_factory = dataset_module.getIntermediateheterFusionDataset


def factory(base_class):
    parent = original_factory(base_class)

    class OtherVehiclesOnly(parent):
        def __getitem__(self, index):
            result = super().__getitem__(index)
            if result is not None and self.params.get("_qcar_exclude_reference_ego", False):
                radius = float(self.params["_qcar_ego_exclusion_radius_model"])
                other_vehicle_targets(result["ego"], self.anchor_box,
                                      self.post_processor.generate_label, radius)
            return result

        def post_process(self, data, predictions):
            if not self.params.get("_qcar_exclude_reference_ego", False):
                return super().post_process(data, predictions)
            # Exclude own-body proposals BEFORE NMS so they cannot suppress
            # valid nearby cars; the context leaves other dataset calls intact.
            token = active_radius.set(float(self.params["_qcar_ego_exclusion_radius_model"]))
            try:
                return super().post_process(data, predictions)
            finally:
                active_radius.reset(token)

    return OtherVehiclesOnly


dataset_module.getIntermediateheterFusionDataset = factory
print("[exclude_reference_ego] opt-in fused target/prediction exclusion; raw datasets unchanged")
