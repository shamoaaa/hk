# Implementation Plan: Audio Augmentation Scene Diversity

## Overview

This plan implements the "analyze first, then augment" principle: analysis tools are built first as prerequisites, then core engine components, then augmentation enhancements, then verification/comparison tools, and finally general utility tools. All new features are opt-in via configuration, preserving full backward compatibility. The implementation language is Python, matching the existing codebase.

## Tasks

- [x] 1. Build analysis toolkit — Phase 1: Analyze (prerequisites)
  - [x] 1.1 Create `07_augment/tools/analyze_real_data.py` — real data acoustic feature extraction
    - Implement `AcousticFeatures` and `AcousticBaseline` dataclasses (with `DistStats`)
    - Extract per-file features: sample rate, duration, RMS energy, peak level, spectral centroid, estimated SNR, spectral envelope
    - Compute aggregate `AcousticBaseline` statistics (mean, std, p5/p25/p50/p75/p95 for each feature)
    - Output `acoustic_baseline.txt` (human-readable) and `acoustic_baseline.json` (machine-readable)
    - Support `--sample_n` for random sampling mode to avoid processing all 200k files
    - Support `--input_dir` pointing to `s10_data_reflow/labeled/` or any wav directory
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_

  - [ ]* 1.2 Write property test for acoustic baseline output completeness
    - **Property 26: Acoustic baseline output completeness**
    - **Validates: Requirements 14.2, 14.3**

  - [x] 1.3 Create `07_augment/tools/recommend_params.py` — baseline-driven parameter recommendation
    - Read `acoustic_baseline.json` as input
    - Derive `snr_range` from baseline SNR distribution (p25-p75 or p5-p95)
    - Derive `gain_range` from baseline RMS energy distribution
    - Derive `eq_gain_range` from baseline spectral envelope variance
    - Output recommended parameters as YAML snippet to stdout
    - Output reasoning for each recommendation (how derived from baseline)
    - Support `--apply <config.yaml>` to merge recommendations into existing config (fill unset params only)
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5_

  - [ ]* 1.4 Write property test for parameter recommendation derivation
    - **Property 27: Parameter recommendation derivation from baseline**
    - **Validates: Requirements 15.1, 15.2**

- [x] 2. Checkpoint — Ensure analysis tools work correctly
  - Ensure all tests pass, ask the user if questions arise.

- [x] 3. Build core engine components — data models and new modules
  - [x] 3.1 Create `07_augment/scene_resolver.py` — SceneResolver class
    - Implement `ResolvedScene` dataclass with all fields (preset_name, scenario_type, chain, main_distance_factor, bg_clarity_factor, strategy_overrides, overlap_mode)
    - Implement `SceneResolver.__init__()` to parse scene_presets, speaking_scenarios, chain_variants from config
    - Implement `SceneResolver.validate_config()` for fail-fast validation (invalid preset names, invalid modes, weights not summing to 1.0, bg_clarity_range min > max)
    - Implement `BUILTIN_PRESETS` dict with living_room, bedroom, study defaults
    - Implement `SceneResolver.resolve()` to produce a `ResolvedScene` for each sample
    - Implement `_resolve_main_distance()` for distance factor → acoustic parameter mapping (RT60, volume, highfreq_cutoff with linear interpolation)
    - Implement scenario-to-chain logic: `single_clean` excludes speaker_mix and noise; `single_noisy` includes noise but excludes speaker_mix; `multi_speaker` includes speaker_mix
    - When no scene/scenario config is present, return passthrough values (backward compat)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2_

  - [ ]* 3.2 Write property tests for SceneResolver
    - **Property 1: Scene preset parameter resolution**
    - **Property 2: Random mode produces valid selections**
    - **Property 4: Invalid preset name causes error**
    - **Property 5: Speaking scenario determines effective chain composition**
    - **Property 7: Main speaker distance factor range invariant**
    - **Property 8: Distance factor monotonic parameter mapping**
    - **Property 9: Distance factor deterministic mapping**
    - **Property 10: Background clarity range constraint**
    - **Validates: Requirements 1.1, 1.3, 1.5, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 4.2**

  - [x] 3.3 Create `07_augment/diversity_controller.py` — DiversityController class
    - Implement `DiversityController.__init__()` with config parsing and counter initialization
    - Implement `next_preset()` with random and balanced modes (balanced: max_count - min_count ≤ 1)
    - Implement `next_scenario()` with balanced and weighted modes
    - Implement `next_chain_variant()` with round_robin and random modes
    - Implement `select_bg_speaker()` with diversity tracking (unique-before-reuse, reset after exhaustion, bg_max_reuse limit)
    - Implement `get_statistics()` returning `DiversityStatistics` dict
    - _Requirements: 1.3, 1.4, 2.5, 2.6, 7.1, 7.2, 7.3, 7.4, 10.1, 10.3, 10.4_

  - [ ]* 3.4 Write property tests for DiversityController
    - **Property 3: Balanced mode distributes evenly**
    - **Property 6: Weighted scenario mode approximates target weights**
    - **Property 13: Background speaker diversity — unique before reuse**
    - **Property 14: Background speaker diversity reset**
    - **Property 15: Background speaker max reuse limit**
    - **Property 21: Chain variant selection validity**
    - **Property 22: Round-robin chain variant ordering**
    - **Validates: Requirements 1.4, 2.5, 2.6, 7.1, 7.2, 7.3, 7.4, 10.1, 10.3**

  - [x] 3.5 Create `07_augment/param_logger.py` — ParamLogger class
    - Implement `ParamLogger.__init__()` with output path and enabled flag
    - Implement `ParamLogger.log()` to write `AugmentParamRecord` as JSON line
    - Implement `ParamLogger.close()` to flush and close file handle
    - When `enabled=False`, all methods are no-ops
    - _Requirements: 13.1, 13.2_

  - [ ]* 3.6 Write property test for ParamLogger
    - **Property 24: Parameter logging accuracy**
    - **Validates: Requirements 13.1, 13.2**

  - [x] 3.7 Create `07_augment/diversity_reporter.py` — DiversityReporter class
    - Implement `DiversityReporter.__init__()` accepting statistics dict
    - Implement `write_text_report()` outputting human-readable `diversity_report.txt`
    - Implement `write_json_report()` outputting machine-readable `diversity_summary.json`
    - Implement `check_imbalance()` with 60% threshold warning logic
    - Report must contain: preset distribution, scenario distribution, overlap_mode distribution, bg_clarity histogram, main_distance histogram, chain_variant distribution
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [ ]* 3.8 Write property tests for DiversityReporter
    - **Property 18: Diversity report contains all required dimensions**
    - **Property 19: JSON report validity and round-trip**
    - **Property 20: Imbalance warning threshold**
    - **Validates: Requirements 9.2, 9.4, 9.5**

- [x] 4. Checkpoint — Ensure all core components pass tests
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Integrate core components into augmentation engine
  - [x] 5.1 Extend `07_augment/augment_engine.py` — wire SceneResolver, DiversityController, ParamLogger
    - Import and instantiate `SceneResolver` in `AudioAugmentor.__init__()`, call `validate_config()` for fail-fast
    - Add `init_batch()` method: create `DiversityController` and `ParamLogger` per batch
    - Add `finalize_batch()` method: close logger, generate diversity report via `DiversityReporter`
    - Modify `augment_single()` to call `scene_resolver.resolve()` and use `ResolvedScene` for chain selection and strategy overrides
    - Add `_merge_strategies()` helper to apply scene/scenario overrides on top of base strategies
    - Implement mutual exclusion logic for `real_rir` / `reverb` (only first encountered in chain is applied)
    - Implement main speaker distance override: when `main_speaker_distance: true`, use distance factor `d` to drive RT60, volume, highfreq instead of independent sampling
    - Implement `bg_clarity_range` constraint: clamp background speaker distance factor `t` to configured range
    - Implement noise category weights support in `noise_from_file` strategy (subdirectory-based categories)
    - Log actual parameters via `ParamLogger` after each augmentation
    - Ensure backward compatibility: when no new config fields are present, behavior is identical to current version
    - _Requirements: 1.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 6.1, 6.2, 6.3, 6.4, 6.5, 10.1, 10.2, 10.3, 10.4, 11.1, 11.2, 11.3, 11.4, 13.1, 13.2_

  - [ ]* 5.2 Write property test for mutual exclusion of reverb strategies
    - **Property 11: Mutual exclusion of reverb strategies**
    - **Validates: Requirements 6.3**

  - [ ]* 5.3 Write property test for peak level preservation after RIR convolution
    - **Property 12: Peak level preservation after RIR convolution**
    - **Validates: Requirements 6.4**

  - [ ]* 5.4 Write property test for preset overrides strategy on conflict
    - **Property 23: Preset overrides strategy on conflict**
    - **Validates: Requirements 11.4**

  - [x] 5.5 Extend `07_augment/augment_dir.py` — background speaker source separation and batch lifecycle
    - Add `--bg_speaker_dir` CLI argument
    - When `bg_speaker_dir` is specified: scan and preload bg wav files into separate `bg_audio_cache`, pass as `audio_cache` and `candidate_paths` to `augment_single()`
    - When `bg_speaker_dir` is not specified: use main speaker pool as background candidates (current behavior)
    - Call `augmentor.init_batch()` before processing loop
    - Call `augmentor.finalize_batch()` after processing loop
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

  - [x] 5.6 Extend `07_augment/augment_manifest.py` — background speaker source separation and batch lifecycle
    - Add `--bg_speaker_dir` CLI argument
    - Mirror the same bg_speaker_dir logic as augment_dir.py (separate bg cache when specified, self-mix when not)
    - Call `augmentor.init_batch()` before processing loop
    - Call `augmentor.finalize_batch()` after processing loop
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

  - [ ]* 5.7 Write property tests for background speaker source isolation and self-mix compatibility
    - **Property 16: Background speaker source isolation**
    - **Property 17: Self-mix mode backward compatibility**
    - **Validates: Requirements 8.1, 8.2, 8.4**

- [x] 6. Checkpoint — Ensure engine integration works end-to-end
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Build verification tool — Phase 4: Verify (post-augmentation)
  - [x] 7.1 Create `07_augment/tools/compare_distributions.py` — real vs augmented distribution comparison
    - Accept `--baseline <acoustic_baseline.json>` and `--augmented_dir <path>` arguments
    - Extract acoustic features from augmented data (reuse feature extraction logic from analyze_real_data.py)
    - Compute KL divergence per feature dimension between baseline and augmented distributions
    - Flag features where KL divergence exceeds configurable threshold with warning and adjustment suggestion
    - Output text report and comparison charts (matplotlib)
    - _Requirements: 16.1, 16.2, 16.3, 16.4_

  - [ ]* 7.2 Write property test for distribution comparison divergence detection
    - **Property 28: Distribution comparison detects significant divergence**
    - **Validates: Requirements 16.2**

- [x] 8. Build general analysis and utility tools
  - [x] 8.1 Create `07_augment/tools/plot_spectrogram.py` — spectrogram visualization
    - Support single-file Mel and linear spectrogram generation
    - Support side-by-side comparison when both original and augmented paths are given
    - Support batch mode with `--input_dir` for all wav files in a directory
    - Annotate spectrograms with sample rate, duration, peak level, RMS energy
    - Support CLI params for resolution, frequency range, colormap
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

  - [x] 8.2 Create `07_augment/tools/print_augment_params.py` — parameter query and audit
    - Read `augment_params.jsonl` file
    - Support query by filename, filter by parameter range
    - Support `--summary` flag for statistical summary (mean, std, min, max, quartiles) per dimension
    - _Requirements: 13.3, 13.5_

  - [x] 8.3 Create `07_augment/tools/plot_param_distribution.py` — parameter distribution visualization
    - Read `augment_params.jsonl` file
    - Generate histograms per parameter dimension
    - Generate scatter plots for joint distributions (e.g., SNR vs RT60)
    - _Requirements: 13.4_

  - [x] 8.4 Create `07_augment/tools/quick_augment.py` — single-file quick augmentation
    - Accept single audio file + config path, output augmented audio
    - Support multiple config files to produce multiple augmented versions
    - Support `--override key=value` for temporary parameter overrides without modifying config
    - _Requirements: 17.1, 17.2, 17.3_

  - [ ]* 8.5 Write property test for CLI parameter override effectiveness
    - **Property 25: CLI parameter override effectiveness**
    - **Validates: Requirements 17.3**

  - [x] 8.6 Create `07_augment/tools/generate_ab_report.py` — HTML A/B comparison page
    - Generate HTML page with original and augmented audio players
    - Include spectrogram thumbnails and key parameter info per audio
    - Support multiple augmented versions for side-by-side comparison
    - _Requirements: 17.4, 17.5_

- [x] 9. Checkpoint — Ensure all tools and tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Config schema extension and backward compatibility verification
  - [x] 10.1 Create example config file with all new fields documented
    - Add all new optional fields (scene_presets, scene_mode, speaking_scenarios, scenario_mode, scenario_weights, main_speaker_distance, bg_clarity_range, bg_speaker_diversity, bg_max_reuse, bg_speaker_dir, chain_variants, chain_variant_mode, noise_category_weights, log_augment_params) with inline comments
    - Verify all fields default to None/false/disabled when omitted
    - _Requirements: 11.1, 11.2_

  - [ ]* 10.2 Write backward compatibility unit tests
    - Load existing configs (`augment_config_speaker.yaml`, `augment_config_speaker_multi.yaml`, `augment_config_speaker_multi_2.yaml`) and verify engine initializes without error
    - Verify that with same seed and no new fields, output matches current behavior
    - _Requirements: 11.1, 11.2, 11.3_

- [x] 11. Dev log and version control setup
  - [x] 11.1 Create `DEVLOG.md` in project root
    - Initialize with project overview and first entry documenting the feature implementation plan
    - Use reverse-chronological ordering (newest first)
    - _Requirements: 18.3, 18.4_

  - [x] 11.2 Create or update `.gitignore`
    - Exclude large audio data files (*.wav in data dirs), temporary outputs, Python __pycache__, .pyc files, augmented output directories
    - _Requirements: 18.5_

- [x] 12. Final checkpoint — Ensure all tests pass and artifacts are complete
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document (28 properties total)
- The task ordering follows the "analyze first, then augment" principle: analysis tools (Phase 1/2) → core engine components → engine integration → verification tools (Phase 4) → general utilities
- All new config fields are opt-in with safe defaults, ensuring zero impact on existing workflows
