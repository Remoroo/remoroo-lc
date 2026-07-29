# G0 scoreboard: `single_6dof_leg` -- **PASS**

- joints: 6  |  TCPs: 1  |  action dim: 6  |  effector widths: [0]
- constraint rows: 62 (50 collision pairs)
- backend: reference

## Bars

| metric | value | bar | |
| --- | ---: | ---: | :-: |
| clean_rms_pos_mm | 0.8235 | 3 | ok |
| clean_rms_rot_deg | 2.405e-05 | 1.5 | ok |
| hard_collisions | 0 | 0 | ok |
| joint_limit_violations | 0 | 0 | ok |
| clean_wall_active_fraction | 0 | 0.05 | ok |

## Tracking (clean tapes, wall-active ticks excluded)

| tape | peak speed | RMS pos | max pos | RMS rot | max rot | ticks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| figure_eight_40mm_5cms | 0.033 m/s | 0.35 mm | 0.54 mm | 0.000 deg | 0.000 deg | 2551 |
| figure_eight_40mm_12cms | 0.079 m/s | 0.82 mm | 1.30 mm | 0.000 deg | 0.000 deg | 1047 |
| figure_eight_80mm_5cms | 0.017 m/s | 0.18 mm | 0.27 mm | 0.000 deg | 0.000 deg | 5127 |
| figure_eight_80mm_12cms | 0.040 m/s | 0.42 mm | 0.65 mm | 0.000 deg | 0.000 deg | 2119 |
| figure_eight_150mm_5cms | 0.009 m/s | 0.09 mm | 0.15 mm | 0.000 deg | 0.000 deg | 9623 |
| figure_eight_150mm_12cms | 0.021 m/s | 0.22 mm | 0.35 mm | 0.000 deg | 0.000 deg | 3991 |
| approach_retreat_120mm | 0.059 m/s | 0.81 mm | 0.95 mm | 0.000 deg | 0.000 deg | 1799 |

## Safety and filter behaviour (all tapes)

| tape | category | min pair | hard coll | limit viol | wall active | max viol | box ticks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| figure_eight_40mm_5cms | clean | 41.0 mm | 0 | 0 | 0.0% | -0.573 | 0 |
| figure_eight_40mm_12cms | clean | 40.8 mm | 0 | 0 | 0.0% | -0.546 | 0 |
| figure_eight_80mm_5cms | clean | 41.0 mm | 0 | 0 | 0.0% | -0.576 | 0 |
| figure_eight_80mm_12cms | clean | 41.0 mm | 0 | 0 | 0.0% | -0.57 | 0 |
| figure_eight_150mm_5cms | clean | 41.0 mm | 0 | 0 | 0.0% | -0.578 | 0 |
| figure_eight_150mm_12cms | clean | 41.0 mm | 0 | 0 | 0.0% | -0.575 | 0 |
| approach_retreat_120mm | clean | 43.5 mm | 0 | 0 | 0.0% | -0.455 | 0 |
| through_the_floor | adversarial | 15.0 mm | 0 | 0 | 91.2% | 0.000803 | 75 |
| into_joint_limits | adversarial | 35.0 mm | 0 | 0 | 1.6% | 0.00079 | 265 |
| beyond_reach | adversarial | 40.3 mm | 0 | 0 | 0.0% | -0.553 | 903 |
| singularity_sweep | singularity | 14.5 mm | 0 | 0 | 77.1% | 0.0118 | 1346 |

## Smoothness and cost

| tape | jerk p50 | jerk p99 | chatter | step p50 | step p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| figure_eight_40mm_5cms | 0.7 | 11.5 | 0.0 /s | 0.349 ms | 0.382 ms |
| figure_eight_40mm_12cms | 4.1 | 20.7 | 0.0 /s | 0.349 ms | 0.380 ms |
| figure_eight_80mm_5cms | 0.2 | 5.9 | 0.0 /s | 0.348 ms | 0.379 ms |
| figure_eight_80mm_12cms | 1.0 | 13.0 | 0.0 /s | 0.349 ms | 0.386 ms |
| figure_eight_150mm_5cms | 0.1 | 3.3 | 0.0 /s | 0.353 ms | 0.400 ms |
| figure_eight_150mm_12cms | 0.3 | 7.5 | 0.0 /s | 0.350 ms | 0.386 ms |
| approach_retreat_120mm | 0.2 | 18.0 | 0.0 /s | 0.353 ms | 0.416 ms |
| through_the_floor | 11.4 | 38.2 | 38.3 /s | 0.351 ms | 0.411 ms |
| into_joint_limits | 1.7 | 58.1 | 2.5 /s | 0.349 ms | 0.384 ms |
| beyond_reach | 1.3 | 904.1 | 0.0 /s | 0.313 ms | 0.355 ms |
| singularity_sweep | 6.3 | 175.5 | 67.4 /s | 0.386 ms | 0.467 ms |

Jerk is rad/s^3 at the joint, from the commanded q_dot.  Chatter counts joint-acceleration sign changes per second while a wall is binding.  Step times are the pure-NumPy reference, which is a specification and not the runtime; see bench.py for the kernels.
