# Linear Scan Threshold Tuning Notes

## 1. What is the linear scan level?

In our H2O2RAM-style hierarchical ORAM, the lowest level is a linear scan buffer.
Every ORAM access reads a block, updates its metadata such as `last_qid`, and
then appends the updated block into this lowest level. When the lowest level is
full, the ORAM triggers a flush and cascade rebuild into higher OHash levels.

In code, this lowest-level capacity is controlled by:

```cpp
buffer_capacity_
```

The original version used:

```cpp
buffer_capacity_ = choose_linear_scan_threshold(sizeof(OramBlock));
```

For our current `OramBlock` size, this selected an effective threshold of 256.

## 2. Why tune the threshold?

Profiling showed that query latency is dominated by ORAM rebuilds, especially:

```text
OHashBucket::build
OHashBucket::extract
occasional OHashTiers::build
```

The linear scan threshold controls how often the lowest level overflows. A larger
linear level means:

```text
larger write buffer
fewer flushes
fewer cascade rebuilds
fewer OHash build/extract operations
shallower hierarchy
```

The cost is that linear buffer lookup scans a larger buffer, so buffer scan cost
can increase. In our experiments, reducing rebuild frequency is more valuable
than the extra scan cost.

## 3. Relation to H2O2RAM

This is consistent with the implementation style of H2O2RAM. H2O2RAM does not
need to start the hierarchy from capacity 2 hash tables. Very small levels can
be more efficiently served by one combined linear scan level. Therefore, the
linear scan threshold is an implementation-level public parameter, not a change
to the abstract ORAM access algorithm.

## 4. Code change

The conservative change is only in `HierarchicalOram::init()`.

Original:

```cpp
buffer_capacity_ = choose_linear_scan_threshold(sizeof(OramBlock));
```

Tuned version:

```cpp
const size_t linear_scan_threshold =
    std::max<size_t>(choose_linear_scan_threshold(sizeof(OramBlock)), 4096);

buffer_capacity_ = linear_scan_threshold;
```

For a 2048 experiment, use:

```cpp
const size_t linear_scan_threshold =
    std::max<size_t>(choose_linear_scan_threshold(sizeof(OramBlock)), 2048);
```

For a 1024 experiment, use:

```cpp
const size_t linear_scan_threshold =
    std::max<size_t>(choose_linear_scan_threshold(sizeof(OramBlock)), 1024);
```

The tiny-layer branch should also use the same `linear_scan_threshold` variable
instead of calling `choose_linear_scan_threshold(...)` again. Otherwise small
test layers may silently fall back to the old threshold.

## 5. What we did not change

We abandoned the query-local batch rebuild idea for now. The current threshold
experiments keep the original update semantics:

```text
access_and_mark()
  -> find_newest()
  -> update last_qid
  -> insert_updated()
  -> if buffer is full: flush_buffer()
  -> push_to_level()
```

There is no query-local delayed commit and no public-size batch rebuild in the
current threshold-only version.

This is important because batch rebuild experiments showed two problems:

```text
1. simple batching can break correctness if an oversized batch is built into an
   OHash table whose capacity is too small;

2. public target-level rebuild restores most correctness but creates very large
   rebuilds and worsens latency.
```

Therefore, the current safe optimization is only to increase the linear scan
threshold.

## 6. Observed hierarchy sizes

The base linear scan threshold determines the number of ORAM levels because each
level doubles in capacity.

| Threshold | Layer 0 max levels | Layer 1 max levels |
|---:|---:|---:|
| 256 | 13 | 7 |
| 1024 | 11 | 5 |
| 2048 | 10 | 4 |
| 4096 | 9 | 3 |

This matches the expected capacity calculation:

```text
Layer 0:
  256  * 2^12 = 1,048,576
  1024 * 2^10 = 1,048,576
  2048 * 2^9  = 1,048,576
  4096 * 2^8  = 1,048,576

Layer 1:
  256  * 2^6 = 16,384
  1024 * 2^4 = 16,384
  2048 * 2^3 = 16,384
  4096 * 2^2 = 16,384
```

## 7. Correctness results

| Configuration | Recall@10 | MRR@10 | Top-k overlap |
|---|---:|---:|---:|
| 256 timing | 0.868 | 1.000 | 0.998 |
| 1024 | 0.867 | 1.000 | 0.997 |
| 2048 | 0.868 | 1.000 | 0.992 |
| 2048 timing | 0.868 | 1.000 | 0.992 |
| 4096 timing | 0.866 | 1.000 | 0.998 |

The threshold-only optimization does not show the severe correctness loss seen
in batch rebuild. The 4096 run has a small Recall@10 drop to 0.866, but MRR
stays 1.0 and Top-k overlap is 0.998. This should be rerun to confirm whether it
is noise or a stable difference.

## 8. Latency results

Timing-enabled runs include `ocall_oram_now_ns` profiling overhead. Non-timing
runs do not. Therefore, compare timing-to-timing and non-timing-to-non-timing.

| Configuration | Avg ms | P50 ms | P95 ms | P99 ms |
|---|---:|---:|---:|---:|
| 256 timing | 1967.64 | 916.84 | 4558.14 | 20358.68 |
| 1024 | 1745.62 | 694.85 | 4340.28 | 20106.58 |
| 2048 | 1623.19 | 614.78 | 4208.37 | 19388.89 |
| 2048 timing | 1700.95 | 699.68 | 4292.34 | 19411.75 |
| 4096 timing | 1549.18 | 365.54 | 4250.10 | 19463.93 |

The most visible improvement is in average and median latency. Tail latency
improves much less.

## 9. Rebuild frequency

| Threshold | L0 build/query | L0 extract/query | L1 build/query | L1 extract/query |
|---:|---:|---:|---:|---:|
| 256 | 5.616 | 5.596 | 3.444 | 3.444 |
| 1024 | 1.404 | 1.384 | 0.869 | 0.848 |
| 2048 | 0.707 | 0.677 | 0.434 | 0.414 |
| 4096 | 0.354 | 0.323 | 0.212 | 0.202 |

Every doubling roughly halves the rebuild frequency. This confirms that the
threshold is doing what we expect.

## 10. Timing breakdown

Timing-enabled OHash cost per query:

| Configuration | Lookup ms | Build ms | Extract ms | Total ms |
|---|---:|---:|---:|---:|
| 256 timing | 228.85 | 1384.70 | 254.78 | 1868.33 |
| 2048 timing | 187.07 | 1220.83 | 211.42 | 1619.32 |
| 4096 timing | 164.43 | 1114.04 | 184.08 | 1462.55 |

Build remains the dominant cost even at threshold 4096. The large remaining
bottlenecks are:

```text
Layer 0 OHashBucket::build
Layer 0 occasional OHashTiers::build
Layer 1 OHashBucket::build
```

## 11. Current recommendation

Use one of the following:

```text
2048:
  more conservative
  Recall@10 remains 0.868
  p99 slightly better than 4096 in current runs

4096:
  best average and median latency
  Recall@10 was 0.866 in one timing run
  p99 does not improve over 2048
```

For a paper/report, 2048 is the safer default. For a performance-focused
prototype run, 4096 is currently the fastest for average and median latency.

## 12. Next step

Increasing the linear scan threshold reduces rebuild frequency, but it does not
remove the cost of large OHash rebuilds. The next meaningful optimization should
focus on:

```text
OHashBucket::build
OHashBucket::extract
large Layer 0 OHashTiers::build
```

The p99 tail remains high because occasional large rebuilds are still expensive.
