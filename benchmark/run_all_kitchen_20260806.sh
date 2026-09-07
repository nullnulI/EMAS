#!/usr/bin/env bash

set -uo pipefail

project_root="/225010231/mwl/EMAS"
result_dir="${project_root}/benchmark/kitchen_results/emas_all_kitchen_qwen_only_20260806"
python_bin="/225010231/miniconda3/envs/conceptgraph/bin/python"

mkdir -p "${result_dir}"
cd "${project_root}" || exit 1

printf '%s\n' "$$" > "${result_dir}/worker.pid"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "${result_dir}/started_at.txt"

env -u __EGL_VENDOR_LIBRARY_DIRS \
  LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
  EMAS_LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
  EMAS_PLANNING_MEMORY_CUDA_VISIBLE_DEVICES=0,1 \
  EMAS_PLANNING_DEVICE=cuda:1 \
  EMAS_PLANNING_GPU_MAX_MEMORY=34GiB \
  EMAS_AGENTS_CUDA_VISIBLE_DEVICES=1 \
  GSA_PATH=/225010231/mwl/EMAS/memory/Grounded-Segment-Anything \
  PYTHONUNBUFFERED=1 \
  "${python_bin}" -u -m benchmark.batch_runner \
    --kitchen-only \
    --mapthor-floorplan-indices 0 \
    --benchmark-seeds 0 \
    --agentnum 2 \
    --platform cloud \
    --class-set scene \
    --execution-mode task_service \
    --task-service-autostart \
    --reuse-planning-model \
    --reuse-task-service \
    --task-service-python /225010231/mwl/Linhao/.homes/userlh/.conda/envs/qwen35/bin/python \
    --task-service-model-path /225010231/mwl/Linhao/models/Qwen3.5-4B \
    --task-service-device cuda \
    --task-service-device-map auto \
    --task-service-dtype float16 \
    --task-service-cuda-visible-devices 1 \
    --task-service-port-base 18090 \
    --receiver-port-base 19010 \
    --qwen-model-path /225010231/mwl/EMAS/Qwen2.5-VL-7B-Instruct \
    --planning-model-path /225010231/mwl/EMAS/Qwen3.5-9B \
    --qwen-num-gpus 1 \
    --planning-max-new-tokens 2048 \
    --planning-max-attempts 3 \
    --max-task-loops 40 \
    --max-task-retries 6 \
    --output-dir "${result_dir}" \
    > "${result_dir}/batch.log" 2>&1
status=$?

printf '%s\n' "${status}" > "${result_dir}/exit_status.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "${result_dir}/finished_at.txt"
exit "${status}"
