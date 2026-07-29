# G0 scoreboard: `mixed` -- **PASS**

- joints: 13  |  TCPs: 2  |  action dim: 15  |  effector widths: [1, 2]
- constraint rows: 292 (266 collision pairs)
- backend: reference

## Bars

| metric | value | bar | |
| --- | ---: | ---: | :-: |
| clean_rms_pos_mm | 2.532 | 3 | ok |
| clean_rms_rot_deg | 0.00176 | 1.5 | ok |
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
| handover | 0.119 m/s | 2.53 mm | 5.52 mm | 0.002 deg | 0.004 deg | 535 |

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
| handover | clean | 57.2 mm | 0 | 0 | 0.0% | -0.931 | 285 |
| through_the_floor | adversarial | 15.4 mm | 0 | 0 | 19.4% | 0.000636 | 442 |
| into_another_chain | adversarial | 15.0 mm | 0 | 0 | 67.0% | 0.000922 | 0 |
| into_joint_limits | adversarial | 19.0 mm | 0 | 0 | 0.0% | -0.0641 | 481 |
| beyond_reach | adversarial | 35.1 mm | 0 | 0 | 7.0% | -0.438 | 737 |
| singularity_sweep | singularity | 17.8 mm | 0 | 0 | 52.8% | -0.0206 | 1478 |

## Smoothness and cost

| tape | jerk p50 | jerk p99 | chatter | step p50 | step p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| figure_eight_40mm_5cms | 1.1 | 17.7 | 0.0 /s | 0.608 ms | 0.674 ms |
| figure_eight_40mm_12cms | 6.5 | 37.0 | 0.0 /s | 0.609 ms | 0.658 ms |
| figure_eight_80mm_5cms | 0.3 | 9.1 | 0.0 /s | 0.609 ms | 0.693 ms |
| figure_eight_80mm_12cms | 1.6 | 20.6 | 0.0 /s | 0.607 ms | 0.706 ms |
| figure_eight_150mm_5cms | 0.1 | 5.3 | 0.0 /s | 0.611 ms | 0.698 ms |
| figure_eight_150mm_12cms | 0.5 | 11.8 | 0.0 /s | 0.610 ms | 0.674 ms |
| approach_retreat_120mm | 0.1 | 16.6 | 0.0 /s | 0.616 ms | 0.679 ms |
| handover | 1.5 | 164.0 | 0.0 /s | 0.588 ms | 0.675 ms |
| through_the_floor | 3.7 | 223.7 | 109.4 /s | 0.625 ms | 1.018 ms |
| into_another_chain | 6.7 | 73.9 | 74.0 /s | 0.704 ms | 0.824 ms |
| into_joint_limits | 15.3 | 152.2 | 0.0 /s | 0.640 ms | 0.717 ms |
| beyond_reach | 1.6 | 48.9 | 42.7 /s | 0.615 ms | 0.655 ms |
| singularity_sweep | 5.8 | 123.1 | 64.1 /s | 0.691 ms | 0.828 ms |

Jerk is rad/s^3 at the joint, from the commanded q_dot.  Chatter counts joint-acceleration sign changes per second while a wall is binding.  Step times are the pure-NumPy reference, which is a specification and not the runtime; see bench.py for the kernels.
