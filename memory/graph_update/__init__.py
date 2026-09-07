"""Scene graph update utilities.

The package is split into observation-driven graph generation, execution-result
relation deltas, and final graph merging.
"""

from .execution_delta import (
    diff_objects,
    infer_relation_deltas_from_metadata,
    infer_relations_for_new_objects,
    index_objects,
)
from .merge import (
    apply_object_deltas,
    apply_relation_deltas,
    append_new_scenegraph_nodes,
    build_ai2thor_to_scenegraph_id_map,
    merge_scenegraph_nodes,
    update_scenegraph_files,
)
from .incremental_observation import (
    mask_known_instances,
    partition_observed_objects,
    prepare_incremental_observation_dataset,
    scenegraph_object_ids,
)
from .observation_update import build_incremental_observed_scenegraph
from .schemas import ObjectDelta, RelationDelta

__all__ = [
    "ObjectDelta",
    "RelationDelta",
    "apply_object_deltas",
    "apply_relation_deltas",
    "append_new_scenegraph_nodes",
    "build_ai2thor_to_scenegraph_id_map",
    "diff_objects",
    "index_objects",
    "infer_relation_deltas_from_metadata",
    "infer_relations_for_new_objects",
    "merge_scenegraph_nodes",
    "build_incremental_observed_scenegraph",
    "mask_known_instances",
    "partition_observed_objects",
    "prepare_incremental_observation_dataset",
    "scenegraph_object_ids",
    "update_scenegraph_files",
]
