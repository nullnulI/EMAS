# Scene Graph Update

这个目录建议负责在线场景图更新，拆成三步：

1. `observation_update.py`
   接收所有 agent 保存下来的 RGB-D/pose/metadata 数据，调用现有 ConceptGraphs pipeline，生成新的观测场景图。

2. `execution_delta.py`
   接收任务执行前后的 AI2-THOR metadata，以及 `agents.skill_plan.execute_allocated_skills` 返回的 trace，根据真实状态差分推导边变化。

3. `merge.py`
   将原始存储场景图、新观测场景图、执行产生的 relation delta 合并，输出新的 `scene_graph.json` 和 `cfslam_object_relations.json`。

## 推荐数据流

```python
from pathlib import Path
from memory.graph_update import infer_relation_deltas_from_metadata, update_scenegraph_files
from memory.graph_update.observation_update import build_observed_scenegraph_from_stage

observed = build_observed_scenegraph_from_stage(post_stage_info, args)

relation_deltas = infer_relation_deltas_from_metadata(
    pre_metadata,
    post_metadata,
    execution=execution_trace,
)

updated = update_scenegraph_files(
    stored_scenegraph_path=Path(current_scenegraph["scene_graph"]),
    stored_relations_path=Path(current_scenegraph["relations"]),
    observed_scenegraph_path=Path(observed["scene_graph"]),
    relation_deltas=relation_deltas,
    output_dir=Path("updated_sg_cache"),
)
```

## 目前支持的边变化

当前先聚焦空间承载关系：

- `a on b`
- `a in b`

边变化来自 AI2-THOR object metadata 的这些字段：

- `parentReceptacles`
- `receptacleObjectIds`
- `isPickedUp`
- `position`
- 以及常见状态字段，如 `isOpen/isToggled/isSliced/isDirty`

生成结果；
remove: Apple a on b CounterTop
add: Apple a in b Bowl
注意：AI2-THOR 返回的是 `objectId`，ConceptGraphs 节点通常是 `id/original_id/pruned_id`。要想边更新特别准，最好在生成场景图节点时额外保存：

```json
{
  "ai2thor_object_id": "Apple|surface|6|28"
}
```

如果暂时没有这个字段，`merge.py` 会尝试从 `ai2thor_object_id/objectId/object_id/name/original_id/id` 里自动建立映射；仍匹配不上时，会保留原始 AI2-THOR objectId 写入 relation，方便后处理排查。

## 观测前增量筛选

不修改原 ConceptGraphs 管线时，可以在 `graph_update` 中先生成一个链接到原
RGB-D 数据的临时数据集。临时数据集只改写 `obj_meta.json`，令 GSA 的
`class_set=scene` prompt 只包含尚未记录的物体类型：

```python
from memory.graph_update import build_incremental_observed_scenegraph

observed = build_incremental_observed_scenegraph(
    dataset_root=Path(stage_info["dataset_root"]),
    scene_id=stage_info["scene_id"],
    cachedir=Path(stage_info["stage_dir"]) / "incremental_sg_cache",
    stored_scenegraph_path=Path(current_scenegraph["scene_graph"]),
    args=args,
    observed_object_ids=visible_object_ids_from_agent_events,
    # 可选：frame filename -> {RGB instance color -> objectId}
    instance_color_maps=instance_color_maps,
)
```

`observed["relation_deltas"]` 只包含 `新物体 -> 已有 receptacle` 的
`a on b/a in b` 关系，可以直接传给 `update_scenegraph_files`。如果提供
`instance_color_maps`，旧实例会在 GSA 运行前从临时 RGB 帧中遮掉；不提供时
只能缩小检测类别，无法排除同类别的旧实例。

## 执行动作关系更新

`generation.py` 的每轮 post-stage 会调用
`infer_relation_deltas_from_metadata(pre_metadata, post_metadata, execution=execution)`：

- 成功 `PickupObject`：删除物体原来的 `a on b/a in b`。
- 成功 `PutObject` 或位置变化：删除旧 parent relation，添加新的 relation。
- 全部低层动作失败：不应用 relation delta。
- `OpenObject/ToggleObject` 等不改变 parent receptacle 的动作不会修改空间边。

每轮推断结果保存到 `merged_scenegraph/relation_deltas.json`，其中分别记录
`execution_deltas`、`new_object_deltas` 和最终应用的 `all_deltas`。

动作造成的节点属性变化保存到 `merged_scenegraph/object_deltas.json`。合并时
按照稳定 `objectId` 对已有节点做字段级更新，保留 caption、视觉特征和其他
场景图专有字段；位置变化还会同步更新节点的 `bbox_center`。AI2-THOR、relay
和 file adapter 都使用相同的“新节点追加、已有节点 patch、关系 delta”流程。
