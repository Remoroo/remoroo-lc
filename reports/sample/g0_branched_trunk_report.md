# G0 scoreboard: `branched_trunk` -- **PASS**

- joints: 11  |  TCPs: 2  |  action dim: 14  |  effector widths: [1, 1]
- constraint rows: 222 (200 collision pairs)
- backend: reference

## Bars

| metric | value | bar | |
| --- | ---: | ---: | :-: |
| clean_rms_pos_mm | 1.829 | 3 | ok |
| clean_rms_rot_deg | 0.01272 | 1.5 | ok |
| hard_collisions | 0 | 0 | ok |
| joint_limit_violations | 0 | 0 | ok |
| clean_wall_active_fraction | 0 | 0.05 | ok |

## Tracking (clean tapes, wall-active ticks excluded)

| tape | peak speed | RMS pos | max pos | RMS rot | max rot | ticks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| figure_eight_40mm_5cms | 0.034 m/s | 0.81 mm | 1.64 mm | 0.006 deg | 0.013 deg | 2551 |
| figure_eight_40mm_12cms | 0.084 m/s | 1.83 mm | 3.56 mm | 0.013 deg | 0.029 deg | 1047 |
| figure_eight_80mm_5cms | 0.017 m/s | 0.40 mm | 0.86 mm | 0.003 deg | 0.007 deg | 5127 |
| figure_eight_80mm_12cms | 0.042 m/s | 0.95 mm | 2.02 mm | 0.007 deg | 0.015 deg | 2119 |
| figure_eight_150mm_5cms | 0.009 m/s | 0.21 mm | 0.46 mm | 0.001 deg | 0.003 deg | 9623 |
| figure_eight_150mm_12cms | 0.022 m/s | 0.51 mm | 1.09 mm | 0.004 deg | 0.008 deg | 3991 |
| approach_retreat_120mm | 0.113 m/s | 1.47 mm | 1.84 mm | 0.000 deg | 0.000 deg | 1799 |

## Safety and filter behaviour (all tapes)

| tape | category | min pair | hard coll | limit viol | wall active | max viol | box ticks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| figure_eight_40mm_5cms | clean | 56.2 mm | 0 | 0 | 0.0% | -0.911 | 0 |
| figure_eight_40mm_12cms | clean | 56.4 mm | 0 | 0 | 0.0% | -0.896 | 0 |
| figure_eight_80mm_5cms | clean | 56.1 mm | 0 | 0 | 0.0% | -0.912 | 0 |
| figure_eight_80mm_12cms | clean | 56.2 mm | 0 | 0 | 0.0% | -0.909 | 0 |
| figure_eight_150mm_5cms | clean | 56.1 mm | 0 | 0 | 0.0% | -0.912 | 0 |
| figure_eight_150mm_12cms | clean | 56.1 mm | 0 | 0 | 0.0% | -0.912 | 0 |
| approach_retreat_120mm | clean | 58.2 mm | 0 | 0 | 0.0% | -0.952 | 0 |
| handover | adversarial | 61.5 mm | 0 | 0 | 0.0% | -1e+06 | 0 |
| through_the_floor | adversarial | 22.1 mm | 0 | 0 | 29.4% | 0.000287 | 224 |
| into_another_chain | adversarial | 61.5 mm | 0 | 0 | 0.0% | -1e+06 | 0 |
| into_joint_limits | adversarial | 14.5 mm | 0 | 0 | 42.4% | 0.0123 | 262 |
| beyond_reach | adversarial | 61.5 mm | 0 | 0 | 0.0% | -1e+06 | 289 |
| singularity_sweep | singularity | 14.7 mm | 0 | 0 | 45.6% | 0.00823 | 1498 |

## Smoothness and cost

| tape | jerk p50 | jerk p99 | chatter | step p50 | step p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| figure_eight_40mm_5cms | 1.0 | 23.7 | 0.0 /s | 0.492 ms | 0.582 ms |
| figure_eight_40mm_12cms | 5.5 | 43.1 | 0.0 /s | 0.487 ms | 0.551 ms |
| figure_eight_80mm_5cms | 0.3 | 12.6 | 0.0 /s | 0.495 ms | 0.559 ms |
| figure_eight_80mm_12cms | 1.5 | 26.4 | 0.0 /s | 0.490 ms | 0.547 ms |
| figure_eight_150mm_5cms | 0.1 | 6.8 | 0.0 /s | 0.491 ms | 0.559 ms |
| figure_eight_150mm_12cms | 0.4 | 15.3 | 0.0 /s | 0.490 ms | 0.585 ms |
| approach_retreat_120mm | 1.0 | 26.9 | 0.0 /s | 0.522 ms | 0.600 ms |
| handover | 1.3 | 8.0 | 0.0 /s | 0.475 ms | 0.510 ms |
| through_the_floor | 18.8 | 267.1 | 322.5 /s | 0.550 ms | 0.961 ms |
| into_another_chain | 1.2 | 7.8 | 0.0 /s | 0.480 ms | 0.520 ms |
| into_joint_limits | 17.5 | 198.8 | 53.5 /s | 0.616 ms | 0.692 ms |
| beyond_reach | 0.6 | 86.6 | 0.0 /s | 0.480 ms | 0.517 ms |
| singularity_sweep | 10.6 | 124.8 | 61.8 /s | 0.570 ms | 0.686 ms |

Jerk is rad/s^3 at the joint, from the commanded q_dot.  Chatter counts joint-acceleration sign changes per second while a wall is binding.  Step times are the pure-NumPy reference, which is a specification and not the runtime; see bench.py for the kernels.
