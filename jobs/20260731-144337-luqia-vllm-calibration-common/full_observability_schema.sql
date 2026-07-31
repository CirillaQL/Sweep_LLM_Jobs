-- Complete ClickHouse schema for the current vLLM calibration experiments.
--
-- The batch uploader writes the 15 core tables below after every completed
-- workload segment. The two raw_* tables are optional seven-day debugging
-- sinks and are not populated during normal collection.
--
-- This file is idempotent: it creates the database, tables, and training view
-- only when they do not already exist. It does not drop or alter existing data.

CREATE DATABASE IF NOT EXISTS vllm_observability;


-- 1. Logical experiment.
CREATE TABLE IF NOT EXISTS vllm_observability.experiments
(
    experiment_id       String CODEC(ZSTD(1)),
    pair_group_id       String CODEC(ZSTD(1)),
    dataset_name        LowCardinality(String) CODEC(ZSTD(1)),
    description         String CODEC(ZSTD(3)),
    created_at          DateTime64(3, 'UTC') CODEC(DoubleDelta, ZSTD(1)),
    tags                Map(LowCardinality(String), String) CODEC(ZSTD(3)),
    updated_at          DateTime64(3, 'UTC') CODEC(DoubleDelta, ZSTD(1))
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY experiment_id;


-- 2. One Slurm segment/job and its concrete vLLM configuration.
CREATE TABLE IF NOT EXISTS vllm_observability.jobs
(
    job_id                          String CODEC(ZSTD(1)),
    experiment_id                   String CODEC(ZSTD(1)),
    pair_group_id                   String CODEC(ZSTD(1)),
    variant_id                      LowCardinality(String) CODEC(ZSTD(1)),
    repeat_no                       UInt16,

    slurm_job_id                    String CODEC(ZSTD(1)),
    job_name                        LowCardinality(String) CODEC(ZSTD(1)),
    status                          LowCardinality(String) CODEC(ZSTD(1)),
    started_at                      DateTime64(3, 'UTC') CODEC(DoubleDelta, ZSTD(1)),
    ended_at                        Nullable(DateTime64(3, 'UTC')) CODEC(ZSTD(1)),

    model                           LowCardinality(String) CODEC(ZSTD(1)),
    topology                        LowCardinality(String) CODEC(ZSTD(1)),
    prefill_node                    LowCardinality(String) CODEC(ZSTD(1)),
    decode_node                     LowCardinality(String) CODEC(ZSTD(1)),
    attention_backend               LowCardinality(String) CODEC(ZSTD(1)),
    kv_connector                    LowCardinality(String) CODEC(ZSTD(1)),

    max_num_seqs                    UInt32,
    max_num_batched_tokens          UInt32,
    gpu_memory_utilization          Float32 CODEC(Gorilla, ZSTD(1)),
    tensor_parallel_size            UInt16,

    dvfs_mode                       LowCardinality(String) CODEC(ZSTD(1)),
    manual_frequency_control        Bool,
    scheduler_prediction            Bool,
    policy_variant                  LowCardinality(String) CODEC(ZSTD(1)),

    kv_cache_metrics                Bool,
    kv_cache_metrics_sample         Float32 CODEC(Gorilla, ZSTD(1)),
    logging_iteration_details       Bool,
    collect_detailed_traces         LowCardinality(String) CODEC(ZSTD(1)),
    prefix_caching                  Bool,
    request_id_headers              Bool,

    config                          Map(LowCardinality(String), String) CODEC(ZSTD(3)),
    updated_at                      DateTime64(3, 'UTC') CODEC(DoubleDelta, ZSTD(1))
)
ENGINE = ReplacingMergeTree(updated_at)
PARTITION BY toYYYYMM(started_at)
ORDER BY (experiment_id, job_id);


-- 3. Node, GPU, network, and runtime topology captured once per role/job.
CREATE TABLE IF NOT EXISTS vllm_observability.job_nodes
(
    job_id                  String CODEC(ZSTD(1)),
    captured_at             DateTime64(9, 'UTC') CODEC(DoubleDelta, ZSTD(1)),
    role                    LowCardinality(String) CODEC(ZSTD(1)),
    node_group              LowCardinality(String) CODEC(ZSTD(1)),
    hostname                LowCardinality(String) CODEC(ZSTD(1)),
    node_ip                 String CODEC(ZSTD(1)),
    peer_ip                 String CODEC(ZSTD(1)),
    network_interface       LowCardinality(String) CODEC(ZSTD(1)),
    link_speed_mbps         UInt32,
    link_mtu                UInt32,

    kernel                  LowCardinality(String) CODEC(ZSTD(1)),
    cpu_count               UInt16,
    gpu_name                LowCardinality(String) CODEC(ZSTD(1)),
    gpu_uuid                String CODEC(ZSTD(1)),
    gpu_memory_total_mib    UInt32,
    driver_version          LowCardinality(String) CODEC(ZSTD(1)),
    cuda_version            LowCardinality(String) CODEC(ZSTD(1)),
    vllm_version            LowCardinality(String) CODEC(ZSTD(1)),
    container_image         LowCardinality(String) CODEC(ZSTD(1)),

    node_work_dir           String CODEC(ZSTD(3)),
    runtime_cwd             String CODEC(ZSTD(3)),
    environment             Map(LowCardinality(String), String) CODEC(ZSTD(3))
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(captured_at)
ORDER BY (job_id, role, captured_at);


-- 4. Workload window and its final summary.
CREATE TABLE IF NOT EXISTS vllm_observability.workload_windows
(
    job_id                      String CODEC(ZSTD(1)),
    window_id                   String CODEC(ZSTD(1)),
    workload_name               LowCardinality(String) CODEC(ZSTD(1)),
    action_id                   LowCardinality(String) CODEC(ZSTD(1)),
    action_config               Map(LowCardinality(String), String) CODEC(ZSTD(3)),

    window_start                DateTime64(9, 'UTC') CODEC(DoubleDelta, ZSTD(1)),
    sending_stopped             Nullable(DateTime64(9, 'UTC')) CODEC(ZSTD(1)),
    window_end                  Nullable(DateTime64(9, 'UTC')) CODEC(ZSTD(1)),

    input_len                   UInt32,
    max_tokens                  UInt32,
    request_rate                Float32 CODEC(Gorilla, ZSTD(1)),
    concurrency                 UInt32,
    num_prompts                 UInt32,
    timeout_ms                  UInt32,
    random_seed                 UInt64,

    planned_requests            UInt32,
    completed_requests          UInt32,
    failed_requests             UInt32,
    timeout_requests            UInt32,
    cancelled_requests          UInt32,
    retry_attempts              UInt32,

    client_observed_drain_s     Float32 CODEC(Gorilla, ZSTD(1)),
    actual_send_rps             Float32 CODEC(Gorilla, ZSTD(1)),
    completion_output_tps       Float32 CODEC(Gorilla, ZSTD(1)),
    mean_ttft_ms                Float32 CODEC(Gorilla, ZSTD(1)),
    p50_ttft_ms                 Float32 CODEC(Gorilla, ZSTD(1)),
    p95_ttft_ms                 Float32 CODEC(Gorilla, ZSTD(1)),
    p99_ttft_ms                 Float32 CODEC(Gorilla, ZSTD(1)),
    mean_tpot_ms                Float32 CODEC(Gorilla, ZSTD(1)),
    p99_tpot_ms                 Float32 CODEC(Gorilla, ZSTD(1)),
    mean_e2e_ms                 Float32 CODEC(Gorilla, ZSTD(1)),
    p99_e2e_ms                  Float32 CODEC(Gorilla, ZSTD(1)),

    updated_at                  DateTime64(3, 'UTC') CODEC(DoubleDelta, ZSTD(1))
)
ENGINE = ReplacingMergeTree(updated_at)
PARTITION BY toYYYYMM(window_start)
ORDER BY (job_id, window_id);


-- 5. Final request record with queue state denormalized at arrival time.
CREATE TABLE IF NOT EXISTS vllm_observability.requests
(
    job_id                          String CODEC(ZSTD(1)),
    window_id                       String CODEC(ZSTD(1)),
    request_index                   UInt32,
    request_id                      String CODEC(ZSTD(1)),
    response_request_id             String CODEC(ZSTD(1)),
    trace_id                        FixedString(32) CODEC(ZSTD(1)),
    client_span_id                  FixedString(16) CODEC(ZSTD(1)),

    planned_send_time               DateTime64(9, 'UTC') CODEC(DoubleDelta, ZSTD(1)),
    client_ready_time               Nullable(DateTime64(9, 'UTC')) CODEC(ZSTD(1)),
    actual_send_time                Nullable(DateTime64(9, 'UTC')) CODEC(ZSTD(1)),
    proxy_arrival_time              Nullable(DateTime64(9, 'UTC')) CODEC(ZSTD(1)),
    response_headers_time           Nullable(DateTime64(9, 'UTC')) CODEC(ZSTD(1)),
    first_token_time                Nullable(DateTime64(9, 'UTC')) CODEC(ZSTD(1)),
    completed_time                  Nullable(DateTime64(9, 'UTC')) CODEC(ZSTD(1)),

    planned_to_ready_ms             Float32 CODEC(Gorilla, ZSTD(1)),
    client_queue_delay_ms           Float32 CODEC(Gorilla, ZSTD(1)),
    send_lag_ms                     Float32 CODEC(Gorilla, ZSTD(1)),
    client_to_proxy_ms              Float32 CODEC(Gorilla, ZSTD(1)),
    ttft_ms                         Float32 CODEC(Gorilla, ZSTD(1)),
    tpot_ms                         Float32 CODEC(Gorilla, ZSTD(1)),
    e2e_ms                          Float32 CODEC(Gorilla, ZSTD(1)),

    planned_input_tokens            UInt32,
    actual_input_tokens             UInt32,
    max_tokens                      UInt32,
    actual_output_tokens            UInt32,
    token_event_count               UInt32,

    http_status                     UInt16,
    outcome                         LowCardinality(String) CODEC(ZSTD(1)),
    timeout                         Bool,
    cancelled                       Bool,
    retry_count                     UInt16,
    attempt_count                   UInt16,
    response_bytes                  UInt64,
    error                           String CODEC(ZSTD(3)),

    prefill_metric_time             Nullable(DateTime64(9, 'UTC')) CODEC(ZSTD(1)),
    prefill_metric_age_ms           Float32 CODEC(Gorilla, ZSTD(1)),
    prefill_running                 UInt32,
    prefill_waiting                 UInt32,
    prefill_waiting_growth_per_s    Float32 CODEC(Gorilla, ZSTD(1)),
    prefill_kv_cache_usage          Float32 CODEC(Gorilla, ZSTD(1)),

    decode_metric_time              Nullable(DateTime64(9, 'UTC')) CODEC(ZSTD(1)),
    decode_metric_age_ms            Float32 CODEC(Gorilla, ZSTD(1)),
    decode_running                  UInt32,
    decode_waiting                  UInt32,
    decode_waiting_growth_per_s     Float32 CODEC(Gorilla, ZSTD(1)),
    decode_kv_cache_usage           Float32 CODEC(Gorilla, ZSTD(1)),

    token_series_stored             Bool,
    token_series_reason             LowCardinality(String) CODEC(ZSTD(1)),

    INDEX idx_request_id request_id TYPE bloom_filter(0.001) GRANULARITY 4,
    INDEX idx_trace_id trace_id TYPE bloom_filter(0.001) GRANULARITY 4
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(planned_send_time)
ORDER BY (job_id, window_id, planned_send_time, request_id);


-- 6. One row per HTTP attempt, including retries, timeouts, and cancellation.
CREATE TABLE IF NOT EXISTS vllm_observability.request_attempts
(
    job_id                  String CODEC(ZSTD(1)),
    window_id               String CODEC(ZSTD(1)),
    request_id              String CODEC(ZSTD(1)),
    attempt                 UInt16,
    started_at              DateTime64(9, 'UTC') CODEC(DoubleDelta, ZSTD(1)),
    response_headers_at     Nullable(DateTime64(9, 'UTC')) CODEC(ZSTD(1)),
    completed_at            Nullable(DateTime64(9, 'UTC')) CODEC(ZSTD(1)),
    http_status             UInt16,
    outcome                 LowCardinality(String) CODEC(ZSTD(1)),
    timeout                 Bool,
    cancelled               Bool,
    retry_scheduled         Bool,
    retry_delay_ms          Float32 CODEC(Gorilla, ZSTD(1)),
    response_bytes          UInt64,
    error                   String CODEC(ZSTD(3)),

    INDEX idx_attempt_request request_id TYPE bloom_filter(0.001) GRANULARITY 4
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(started_at)
ORDER BY (job_id, window_id, request_id, attempt);


-- 7. Unified client/proxy/server request lifecycle events.
CREATE TABLE IF NOT EXISTS vllm_observability.request_events
(
    job_id                  String CODEC(ZSTD(1)),
    window_id               String CODEC(ZSTD(1)),
    request_id              String CODEC(ZSTD(1)),
    internal_request_id     String CODEC(ZSTD(1)),
    trace_id                FixedString(32) CODEC(ZSTD(1)),
    event_time              DateTime64(9, 'UTC') CODEC(DoubleDelta, ZSTD(1)),
    monotonic_ns            UInt64 CODEC(Delta, ZSTD(1)),
    source                  LowCardinality(String) CODEC(ZSTD(1)),
    role                    LowCardinality(String) CODEC(ZSTD(1)),
    stage                   LowCardinality(String) CODEC(ZSTD(1)),
    event                   LowCardinality(String) CODEC(ZSTD(1)),
    attempt                 UInt16,
    route_index             Int16,
    upstream_url            LowCardinality(String) CODEC(ZSTD(1)),
    http_status             UInt16,
    payload_bytes           UInt64,
    detail                  Map(LowCardinality(String), String) CODEC(ZSTD(3)),
    message                 String CODEC(ZSTD(3)),

    INDEX idx_event_request request_id TYPE bloom_filter(0.001) GRANULARITY 4,
    INDEX idx_event_trace trace_id TYPE bloom_filter(0.001) GRANULARITY 4
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(event_time)
ORDER BY (job_id, window_id, request_id, event_time, source, event);


-- 8. Packed per-request token arrival sequence.
CREATE TABLE IF NOT EXISTS vllm_observability.request_token_series
(
    job_id                  String CODEC(ZSTD(1)),
    window_id               String CODEC(ZSTD(1)),
    request_id              String CODEC(ZSTD(1)),
    actual_send_time        DateTime64(9, 'UTC') CODEC(DoubleDelta, ZSTD(1)),
    sample_reason           LowCardinality(String) CODEC(ZSTD(1)),
    token_count             UInt32,
    arrival_offset_us       Array(UInt64) CODEC(ZSTD(3)),
    inter_token_us          Array(UInt32) CODEC(ZSTD(3)),
    fragment_bytes          Array(UInt32) CODEC(ZSTD(3)),
    mean_itl_us             Float32 CODEC(Gorilla, ZSTD(1)),
    p50_itl_us              Float32 CODEC(Gorilla, ZSTD(1)),
    p95_itl_us              Float32 CODEC(Gorilla, ZSTD(1)),
    max_itl_us              UInt64,
    stall_count             UInt32,

    INDEX idx_token_request request_id TYPE bloom_filter(0.001) GRANULARITY 4
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(actual_send_time)
ORDER BY (job_id, window_id, request_id);


-- 9. Required high-frequency engine state, normally one row every 500 ms.
CREATE TABLE IF NOT EXISTS vllm_observability.engine_samples
(
    job_id                          String CODEC(ZSTD(1)),
    event_time                      DateTime64(9, 'UTC') CODEC(DoubleDelta, ZSTD(1)),
    monotonic_ns                    UInt64 CODEC(Delta, ZSTD(1)),
    role                            LowCardinality(String) CODEC(ZSTD(1)),
    endpoint                        LowCardinality(String) CODEC(ZSTD(1)),
    scrape_ok                       Bool,
    scrape_duration_ms              Float32 CODEC(Gorilla, ZSTD(1)),

    running                         UInt32,
    waiting                         UInt32,
    waiting_growth_per_s            Float32 CODEC(Gorilla, ZSTD(1)),
    kv_cache_usage                  Float32 CODEC(Gorilla, ZSTD(1)),

    prefix_cache_hits               UInt64,
    prefix_cache_queries            UInt64,
    external_prefix_cache_hits      UInt64,
    external_prefix_cache_queries   UInt64,
    preemptions                     UInt64,

    prompt_tokens                   UInt64,
    generation_tokens               UInt64,
    successful_requests             UInt64,
    failed_requests                 UInt64,

    e2e_count                       UInt64,
    e2e_sum_s                       Float64 CODEC(Gorilla, ZSTD(1)),
    ttft_count                      UInt64,
    ttft_sum_s                      Float64 CODEC(Gorilla, ZSTD(1)),
    inter_token_count               UInt64,
    inter_token_sum_s               Float64 CODEC(Gorilla, ZSTD(1)),
    iteration_tokens_count          UInt64,
    iteration_tokens_sum            UInt64,
    engine_sleep_state              Int8,

    prompt_tps                      Float32 CODEC(Gorilla, ZSTD(1)),
    generation_tps                  Float32 CODEC(Gorilla, ZSTD(1)),
    completion_rps                  Float32 CODEC(Gorilla, ZSTD(1)),
    error                           String CODEC(ZSTD(3))
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(event_time)
ORDER BY (job_id, role, event_time);


-- 10. Cumulative vLLM histogram snapshots at workload-window boundaries.
CREATE TABLE IF NOT EXISTS vllm_observability.histogram_buckets
(
    job_id              String CODEC(ZSTD(1)),
    window_id           String CODEC(ZSTD(1)),
    captured_at         DateTime64(9, 'UTC') CODEC(DoubleDelta, ZSTD(1)),
    role                LowCardinality(String) CODEC(ZSTD(1)),
    capture_kind        LowCardinality(String) CODEC(ZSTD(1)),
    metric              LowCardinality(String) CODEC(ZSTD(1)),
    labels              Map(LowCardinality(String), String) CODEC(ZSTD(3)),
    bucket_le           Float64 CODEC(Gorilla, ZSTD(1)),
    cumulative_count    UInt64,
    histogram_count     UInt64,
    histogram_sum       Float64 CODEC(Gorilla, ZSTD(1))
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(captured_at)
ORDER BY (job_id, window_id, role, metric, capture_kind, bucket_le);


-- 11. Parsed scheduler and engine iteration details.
CREATE TABLE IF NOT EXISTS vllm_observability.scheduler_iterations
(
    job_id                      String CODEC(ZSTD(1)),
    event_time                  DateTime64(9, 'UTC') CODEC(DoubleDelta, ZSTD(1)),
    role                        LowCardinality(String) CODEC(ZSTD(1)),
    iteration_id                UInt64,
    iteration_duration_us       UInt64,

    running                     UInt32,
    waiting                     UInt32,
    prefill_running             UInt32,
    decode_running              UInt32,
    scheduled_sequences         UInt32,
    scheduled_tokens            UInt32,
    batch_tokens                UInt32,
    preempted_sequences         UInt32,
    finished_sequences          UInt32,
    kv_cache_usage              Float32 CODEC(Gorilla, ZSTD(1)),

    scheduled_request_ids       Array(String) CODEC(ZSTD(3)),
    preempted_request_ids       Array(String) CODEC(ZSTD(3)),
    finished_request_ids        Array(String) CODEC(ZSTD(3)),
    detail                      Map(LowCardinality(String), String) CODEC(ZSTD(3)),
    raw_message                 String CODEC(ZSTD(3))
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(event_time)
ORDER BY (job_id, role, event_time, iteration_id);


-- 12. GPU, DVFS, memory, PCIe, and host-network observations.
CREATE TABLE IF NOT EXISTS vllm_observability.gpu_samples
(
    job_id                      String CODEC(ZSTD(1)),
    event_time                  DateTime64(6, 'UTC') CODEC(DoubleDelta, ZSTD(1)),
    role                        LowCardinality(String) CODEC(ZSTD(1)),
    hostname                    LowCardinality(String) CODEC(ZSTD(1)),
    gpu_index                   UInt8,
    gpu_uuid                    String CODEC(ZSTD(1)),
    network_interface           LowCardinality(String) CODEC(ZSTD(1)),

    rx_bytes                    UInt64 CODEC(Delta, ZSTD(1)),
    tx_bytes                    UInt64 CODEC(Delta, ZSTD(1)),
    rx_bytes_per_s              Float64 CODEC(Gorilla, ZSTD(1)),
    tx_bytes_per_s              Float64 CODEC(Gorilla, ZSTD(1)),

    gpu_util_pct                Float32 CODEC(Gorilla, ZSTD(1)),
    gpu_power_w                 Float32 CODEC(Gorilla, ZSTD(1)),
    gpu_sm_mhz                  UInt32,
    gpu_memory_used_mib         UInt32,
    gpu_mem_util_pct            Float32 CODEC(Gorilla, ZSTD(1)),
    gpu_power_limit_w           Float32 CODEC(Gorilla, ZSTD(1)),
    gpu_mem_clock_mhz           UInt32,
    gpu_temperature_c           Float32 CODEC(Gorilla, ZSTD(1)),
    gpu_memory_total_mib        UInt32,
    gpu_pstate                  LowCardinality(String) CODEC(ZSTD(1)),
    dvfs_mode                   LowCardinality(String) CODEC(ZSTD(1)),
    throttle_reasons            Array(String) CODEC(ZSTD(1)),
    pcie_rx_bytes_per_s         Float64 CODEC(Gorilla, ZSTD(1)),
    pcie_tx_bytes_per_s         Float64 CODEC(Gorilla, ZSTD(1))
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(event_time)
ORDER BY (job_id, role, hostname, gpu_index, event_time);


-- 13. Request-correlated KV transfer observations for PD experiments.
-- Current homogeneous calibration jobs leave this table empty by design.
CREATE TABLE IF NOT EXISTS vllm_observability.kv_transfer_events
(
    job_id                  String CODEC(ZSTD(1)),
    window_id               String CODEC(ZSTD(1)),
    request_id              String CODEC(ZSTD(1)),
    trace_id                FixedString(32) CODEC(ZSTD(1)),
    transfer_id             String CODEC(ZSTD(1)),
    started_at              DateTime64(9, 'UTC') CODEC(DoubleDelta, ZSTD(1)),
    completed_at            Nullable(DateTime64(9, 'UTC')) CODEC(ZSTD(1)),
    source_role             LowCardinality(String) CODEC(ZSTD(1)),
    destination_role        LowCardinality(String) CODEC(ZSTD(1)),
    transport               LowCardinality(String) CODEC(ZSTD(1)),
    status                  LowCardinality(String) CODEC(ZSTD(1)),
    blocks                  UInt32,
    transfer_bytes          UInt64,
    queue_delay_us          UInt64,
    transfer_duration_us    UInt64,
    retry_count             UInt16,
    error                   String CODEC(ZSTD(3)),
    attributes              Map(LowCardinality(String), String) CODEC(ZSTD(3)),

    INDEX idx_kv_request request_id TYPE bloom_filter(0.001) GRANULARITY 4,
    INDEX idx_kv_trace trace_id TYPE bloom_filter(0.001) GRANULARITY 4
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(started_at)
ORDER BY (job_id, window_id, request_id, started_at);


-- 14. OpenTelemetry spans emitted by vLLM and the benchmark client.
CREATE TABLE IF NOT EXISTS vllm_observability.otel_spans
(
    job_id                  String CODEC(ZSTD(1)),
    window_id               String CODEC(ZSTD(1)),
    request_id              String CODEC(ZSTD(1)),
    received_at             DateTime64(9, 'UTC') CODEC(DoubleDelta, ZSTD(1)),
    batch_id                String CODEC(ZSTD(1)),
    service_name            LowCardinality(String) CODEC(ZSTD(1)),
    scope_name              LowCardinality(String) CODEC(ZSTD(1)),
    scope_version           LowCardinality(String) CODEC(ZSTD(1)),
    trace_id                FixedString(32) CODEC(ZSTD(1)),
    span_id                 FixedString(16) CODEC(ZSTD(1)),
    parent_span_id          FixedString(16) CODEC(ZSTD(1)),
    span_name               LowCardinality(String) CODEC(ZSTD(1)),
    span_kind               UInt8,
    start_time              DateTime64(9, 'UTC') CODEC(DoubleDelta, ZSTD(1)),
    end_time                DateTime64(9, 'UTC') CODEC(DoubleDelta, ZSTD(1)),
    duration_us             UInt64,
    status_code             UInt8,
    status_message          String CODEC(ZSTD(3)),
    resource_attributes     Map(LowCardinality(String), String) CODEC(ZSTD(3)),
    span_attributes         Map(LowCardinality(String), String) CODEC(ZSTD(3)),
    events_json             String CODEC(ZSTD(3)),
    links_count             UInt16,

    INDEX idx_span_request request_id TYPE bloom_filter(0.001) GRANULARITY 4,
    INDEX idx_span_trace trace_id TYPE bloom_filter(0.001) GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(start_time)
ORDER BY (job_id, service_name, start_time, trace_id, span_id);


-- 15. Queue drain samples after stopping arrivals.
CREATE TABLE IF NOT EXISTS vllm_observability.drain_samples
(
    job_id              String CODEC(ZSTD(1)),
    window_id           String CODEC(ZSTD(1)),
    event_time          DateTime64(9, 'UTC') CODEC(DoubleDelta, ZSTD(1)),
    monotonic_ns        UInt64 CODEC(Delta, ZSTD(1)),
    elapsed_s           Float32 CODEC(Gorilla, ZSTD(1)),
    role                LowCardinality(String) CODEC(ZSTD(1)),
    running             UInt32,
    waiting             UInt32,
    kv_cache_usage      Float32 CODEC(Gorilla, ZSTD(1)),
    scrape_ok           Bool,
    drained             Bool,
    error               String CODEC(ZSTD(3))
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(event_time)
ORDER BY (job_id, window_id, role, event_time);


-- 16. Optional raw vLLM metrics for collector debugging only.
CREATE TABLE IF NOT EXISTS vllm_observability.raw_metrics_debug
(
    job_id              String CODEC(ZSTD(1)),
    event_time          DateTime64(9, 'UTC') CODEC(DoubleDelta, ZSTD(1)),
    role                LowCardinality(String) CODEC(ZSTD(1)),
    endpoint            LowCardinality(String) CODEC(ZSTD(1)),
    metric              LowCardinality(String) CODEC(ZSTD(1)),
    labels              Map(LowCardinality(String), String) CODEC(ZSTD(3)),
    value               Float64 CODEC(Gorilla, ZSTD(1))
)
ENGINE = MergeTree
PARTITION BY toDate(event_time)
ORDER BY (job_id, role, metric, event_time)
TTL event_time + INTERVAL 7 DAY DELETE;


-- 17. Optional unparsed vLLM log events for parser debugging only.
CREATE TABLE IF NOT EXISTS vllm_observability.raw_engine_log_debug
(
    job_id              String CODEC(ZSTD(1)),
    event_time          DateTime64(9, 'UTC') CODEC(DoubleDelta, ZSTD(1)),
    role                LowCardinality(String) CODEC(ZSTD(1)),
    source_file         LowCardinality(String) CODEC(ZSTD(1)),
    line_number         UInt64,
    category            LowCardinality(String) CODEC(ZSTD(1)),
    request_id          String CODEC(ZSTD(1)),
    message             String CODEC(ZSTD(6)),

    INDEX idx_log_request request_id TYPE bloom_filter(0.001) GRANULARITY 4
)
ENGINE = MergeTree
PARTITION BY toDate(event_time)
ORDER BY (job_id, role, category, event_time, line_number)
TTL event_time + INTERVAL 7 DAY DELETE;


-- Request-level training view with workload/action metadata attached.
CREATE VIEW IF NOT EXISTS vllm_observability.request_training AS
SELECT
    r.*,
    w.workload_name,
    w.action_id,
    w.input_len AS configured_input_len,
    w.max_tokens AS configured_max_tokens,
    w.request_rate,
    w.concurrency,
    w.num_prompts,
    w.action_config
FROM vllm_observability.requests AS r
ANY LEFT JOIN
(
    SELECT *
    FROM vllm_observability.workload_windows FINAL
) AS w
    ON r.job_id = w.job_id
   AND r.window_id = w.window_id;
