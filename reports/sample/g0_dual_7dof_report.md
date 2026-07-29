# G0 scoreboard: `dual_7dof` -- **PASS**

- joints: 14  |  TCPs: 2  |  action dim: 14  |  effector widths: [1, 1]
- constraint rows: 318 (290 collision pairs)
- backend: reference

## Bars

| metric | value | bar | |
| --- | ---: | ---: | :-: |
| clean_rms_pos_mm | 1.364 | 3 | ok |
| clean_rms_rot_deg | 0.0004372 | 1.5 | ok |
| hard_collisions | 0 | 0 | ok |
| joint_limit_violations | 0 | 0 | ok |
| clean_wall_active_fraction | 0 | 0.05 | ok |

## Tracking (clean tapes, wall-active ticks excluded)

| tape | peak speed | RMS pos | max pos | RMS rot | max rot | ticks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| figure_eight_40mm_5cms | 0.042 m/s | 0.45 mm | 0.69 mm | 0.000 deg | 0.000 deg | 2551 |
| figure_eight_40mm_12cms | 0.104 m/s | 1.06 mm | 1.73 mm | 0.000 deg | 0.001 deg | 1047 |
| figure_eight_80mm_5cms | 0.021 m/s | 0.22 mm | 0.35 mm | 0.000 deg | 0.000 deg | 5127 |
| figure_eight_80mm_12cms | 0.051 m/s | 0.54 mm | 0.84 mm | 0.000 deg | 0.000 deg | 2119 |
| figure_eight_150mm_5cms | 0.011 m/s | 0.12 mm | 0.19 mm | 0.000 deg | 0.000 deg | 9623 |
| figure_eight_150mm_12cms | 0.027 m/s | 0.29 mm | 0.45 mm | 0.000 deg | 0.000 deg | 3991 |
| approach_retreat_120mm | 0.032 m/s | 0.48 mm | 0.54 mm | 0.000 deg | 0.000 deg | 1799 |
| handover | 0.108 m/s | 1.36 mm | 1.79 mm | 0.000 deg | 0.001 deg | 535 |

## Safety and filter behaviour (all tapes)

| tape | category | min pair | hard coll | limit viol | wall active | max viol | box ticks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| figure_eight_40mm_5cms | clean | 53.3 mm | 0 | 0 | 0.0% | -0.843 | 0 |
| figure_eight_40mm_12cms | clean | 53.7 mm | 0 | 0 | 0.0% | -0.834 | 0 |
| figure_eight_80mm_5cms | clean | 53.2 mm | 0 | 0 | 0.0% | -0.847 | 0 |
| figure_eight_80mm_12cms | clean | 53.2 mm | 0 | 0 | 0.0% | -0.84 | 0 |
| figure_eight_150mm_5cms | clean | 53.1 mm | 0 | 0 | 0.0% | -0.845 | 0 |
| figure_eight_150mm_12cms | clean | 53.1 mm | 0 | 0 | 0.0% | -0.845 | 0 |
| approach_retreat_120mm | clean | 52.8 mm | 0 | 0 | 0.0% | -0.824 | 0 |
| handover | clean | 55.6 mm | 0 | 0 | 0.0% | -0.89 | 0 |
| through_the_floor | adversarial | 15.4 mm | 0 | 0 | 19.0% | 0.000419 | 441 |
| into_another_chain | adversarial | 15.1 mm | 0 | 0 | 66.8% | 0.000779 | 100 |
| into_joint_limits | adversarial | 18.6 mm | 0 | 0 | 0.0% | -0.0714 | 0 |
| beyond_reach | adversarial | 31.3 mm | 0 | 0 | 7.1% | -0.356 | 369 |
| singularity_sweep | singularity | 34.3 mm | 0 | 0 | 23.4% | 0.000245 | 1313 |

## Smoothness and cost

| tape | jerk p50 | jerk p99 | chatter | step p50 | step p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| figure_eight_40mm_5cms | 1.1 | 17.8 | 0.0 /s | 0.700 ms | 0.771 ms |
| figure_eight_40mm_12cms | 6.5 | 37.0 | 0.0 /s | 0.695 ms | 0.753 ms |
| figure_eight_80mm_5cms | 0.3 | 9.1 | 0.0 /s | 0.699 ms | 0.790 ms |
| figure_eight_80mm_12cms | 1.6 | 20.6 | 0.0 /s | 0.708 ms | 0.811 ms |
| figure_eight_150mm_5cms | 0.1 | 5.3 | 0.0 /s | 0.699 ms | 0.789 ms |
| figure_eight_150mm_12cms | 0.5 | 11.6 | 0.0 /s | 0.703 ms | 0.799 ms |
| approach_retreat_120mm | 0.1 | 16.6 | 0.0 /s | 0.710 ms | 0.786 ms |
| handover | 1.0 | 28.6 | 0.0 /s | 0.691 ms | 0.780 ms |
| through_the_floor | 3.4 | 223.7 | 20.5 /s | 0.736 ms | 1.053 ms |
| into_another_chain | 4.5 | 113.3 | 155.3 /s | 0.812 ms | 0.916 ms |
| into_joint_limits | 4.5 | 50.9 | 0.0 /s | 0.731 ms | 0.830 ms |
| beyond_reach | 2.5 | 53.3 | 24.9 /s | 0.695 ms | 0.815 ms |
| singularity_sweep | 10.0 | 173.3 | 35.0 /s | 0.723 ms | 0.808 ms |

Jerk is rad/s^3 at the joint, from the commanded q_dot.  Chatter counts joint-acceleration sign changes per second while a wall is binding.  Step times are the pure-NumPy reference, which is a specification and not the runtime; see bench.py for the kernels.
