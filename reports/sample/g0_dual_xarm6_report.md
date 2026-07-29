# G0 scoreboard: `dual_xarm6` -- **PASS**

- joints: 12  |  TCPs: 2  |  action dim: 14  |  effector widths: [1, 1]
- constraint rows: 267 (243 collision pairs)
- backend: reference

## Bars

| metric | value | bar | |
| --- | ---: | ---: | :-: |
| clean_rms_pos_mm | 1.811 | 3 | ok |
| clean_rms_rot_deg | 0.001306 | 1.5 | ok |
| hard_collisions | 0 | 0 | ok |
| joint_limit_violations | 0 | 0 | ok |
| clean_wall_active_fraction | 0 | 0.05 | ok |

## Tracking (clean tapes, wall-active ticks excluded)

| tape | peak speed | RMS pos | max pos | RMS rot | max rot | ticks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| figure_eight_40mm_5cms | 0.071 m/s | 0.75 mm | 1.16 mm | 0.000 deg | 0.000 deg | 2551 |
| figure_eight_40mm_12cms | 0.167 m/s | 1.73 mm | 2.75 mm | 0.000 deg | 0.000 deg | 1047 |
| figure_eight_80mm_5cms | 0.062 m/s | 0.66 mm | 1.02 mm | 0.000 deg | 0.000 deg | 5127 |
| figure_eight_80mm_12cms | 0.152 m/s | 1.57 mm | 2.51 mm | 0.000 deg | 0.000 deg | 2119 |
| figure_eight_150mm_5cms | 0.033 m/s | 0.35 mm | 0.54 mm | 0.000 deg | 0.000 deg | 9623 |
| figure_eight_150mm_12cms | 0.080 m/s | 0.84 mm | 1.32 mm | 0.000 deg | 0.000 deg | 3991 |
| approach_retreat_120mm | 0.113 m/s | 1.47 mm | 1.84 mm | 0.000 deg | 0.000 deg | 1799 |
| handover | 0.114 m/s | 1.81 mm | 3.73 mm | 0.001 deg | 0.003 deg | 535 |

## Safety and filter behaviour (all tapes)

| tape | category | min pair | hard coll | limit viol | wall active | max viol | box ticks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| figure_eight_40mm_5cms | clean | 69.8 mm | 0 | 0 | 0.0% | -1e+06 | 0 |
| figure_eight_40mm_12cms | clean | 71.1 mm | 0 | 0 | 0.0% | -1e+06 | 0 |
| figure_eight_80mm_5cms | clean | 62.2 mm | 0 | 0 | 0.0% | -1e+06 | 0 |
| figure_eight_80mm_12cms | clean | 63.0 mm | 0 | 0 | 0.0% | -1e+06 | 0 |
| figure_eight_150mm_5cms | clean | 62.1 mm | 0 | 0 | 0.0% | -1e+06 | 0 |
| figure_eight_150mm_12cms | clean | 62.3 mm | 0 | 0 | 0.0% | -1e+06 | 0 |
| approach_retreat_120mm | clean | 68.7 mm | 0 | 0 | 0.0% | -1e+06 | 0 |
| handover | clean | 72.5 mm | 0 | 0 | 0.0% | -1e+06 | 222 |
| through_the_floor | adversarial | 15.0 mm | 0 | 0 | 31.7% | 0.000636 | 0 |
| into_another_chain | adversarial | 15.0 mm | 0 | 0 | 66.9% | 0.0006 | 0 |
| into_joint_limits | adversarial | 29.5 mm | 0 | 0 | 0.0% | -0.29 | 476 |
| beyond_reach | adversarial | 55.4 mm | 0 | 0 | 0.0% | -0.888 | 543 |
| singularity_sweep | singularity | 19.0 mm | 0 | 0 | 6.1% | -0.0678 | 1520 |

## Smoothness and cost

| tape | jerk p50 | jerk p99 | chatter | step p50 | step p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| figure_eight_40mm_5cms | 1.3 | 21.0 | 0.0 /s | 0.515 ms | 0.586 ms |
| figure_eight_40mm_12cms | 8.3 | 34.6 | 0.0 /s | 0.512 ms | 0.558 ms |
| figure_eight_80mm_5cms | 0.6 | 18.9 | 0.0 /s | 0.516 ms | 0.589 ms |
| figure_eight_80mm_12cms | 3.6 | 32.9 | 0.0 /s | 0.520 ms | 0.783 ms |
| figure_eight_150mm_5cms | 0.2 | 11.3 | 0.0 /s | 0.516 ms | 0.792 ms |
| figure_eight_150mm_12cms | 1.0 | 22.6 | 0.0 /s | 0.515 ms | 0.577 ms |
| approach_retreat_120mm | 0.7 | 26.8 | 0.0 /s | 0.515 ms | 0.549 ms |
| handover | 1.5 | 133.3 | 0.0 /s | 0.517 ms | 0.555 ms |
| through_the_floor | 5.5 | 30.4 | 325.4 /s | 0.518 ms | 0.831 ms |
| into_another_chain | 3.5 | 62.3 | 67.9 /s | 0.652 ms | 0.775 ms |
| into_joint_limits | 14.1 | 148.1 | 0.0 /s | 0.585 ms | 0.650 ms |
| beyond_reach | 1.1 | 37.9 | 0.0 /s | 0.523 ms | 0.603 ms |
| singularity_sweep | 6.3 | 59.2 | 5.0 /s | 0.580 ms | 0.663 ms |

Jerk is rad/s^3 at the joint, from the commanded q_dot.  Chatter counts joint-acceleration sign changes per second while a wall is binding.  Step times are the pure-NumPy reference, which is a specification and not the runtime; see bench.py for the kernels.
