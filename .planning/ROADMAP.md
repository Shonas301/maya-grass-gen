# Roadmap: Maya Grass Plugin v1.0

## Overview

This milestone migrates the Maya grass plugin from the `noise` library to `opensimplex` for Windows compatibility, creates a clean single-function entry point, and verifies that wind animation works correctly in both viewport playback and final renders. The work progresses from low-level noise utilities through module migration, entry point creation, and verification testing.

## Phases

**Phase Numbering:**
- Integer phases (70, 71, 72...): Planned milestone work
- Decimal phases (70.1, 70.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 70: Noise Wrapper Foundation** - Create opensimplex fBm wrapper with matching API
- [ ] **Phase 71: FlowField Migration** - Migrate FlowField from noise.pnoise3 to opensimplex fBm
- [x] **Phase 72: Flow Field Migration** - Migrate flow_field.py and PointClusterer noise dependencies
- [x] **Phase 73: Entry Point Creation** - Create generate_grass() single-function entry point
- [x] **Phase 74: Dependency Cleanup** - Remove noise library, update requirements
- [ ] **Phase 75: Viewport Animation Verification** - Verify wind animation in Maya viewport
- [ ] **Phase 76: Render Animation Verification** - Verify animation in playblast and batch render
- [ ] **Phase 77: Integration Testing** - End-to-end workflow testing
- [ ] **Phase 78: Documentation and Polish** - Usage documentation and edge case handling
- [ ] **Phase 79: Obstacle Proximity Density** - Higher grass density near objects (foot traffic avoidance effect)

## Phase Details

### Phase 70: Noise Wrapper Foundation
**Goal**: Create a drop-in replacement fBm wrapper for opensimplex that matches noise library's API
**Depends on**: Nothing (first phase)
**Requirements**: REQ-001
**Success Criteria** (what must be TRUE):
  1. `fbm_noise2()` accepts octaves, persistence, lacunarity parameters
  2. `fbm_noise3()` accepts octaves, persistence, lacunarity parameters (z for time animation)
  3. `fbm_noise1()` available for 1D noise (implemented via noise2)
  4. `init_noise()` seeds the generator once at initialization
  5. Output range normalized to [-1.0, 1.0]
**Plans**: 2 plans

Plans:
- [x] 70-01-PLAN.md — Create noise_utils.py module with fBm wrapper functions
- [x] 70-02-PLAN.md — Add unit tests for fBm wrapper (octave accumulation, normalization)

---

### Phase 71: FlowField Migration
**Goal**: Migrate FlowField class to use opensimplex-based fBm for wind calculation
**Depends on**: Phase 70
**Requirements**: REQ-002
**Success Criteria** (what must be TRUE):
  1. FlowField uses `fbm_noise3()` for wind vector calculation
  2. Time-based animation works via z parameter (time_offset)
  3. Wind deflection around obstacles preserved (flow field integration)
  4. Visual output comparable to previous (organic wind patterns)
  5. MASH network creates distribute node with correct name resolution
**Plans**: 3 plans

Plans:
- [x] 71-01-PLAN.md — Replace noise.pnoise3 in FlowField.get_base_flow() with fbm_noise3
- [x] 71-02-PLAN.md — Add continuity tests verifying organic wind patterns
- [ ] 71-03-PLAN.md — Fix MASH node naming issue (gap closure from UAT)

---

### Phase 72: Flow Field Migration
**Goal**: Verify flow_field.py migration complete and PointClusterer requires no changes
**Depends on**: Phase 70
**Requirements**: REQ-003
**Success Criteria** (what must be TRUE):
  1. flow_field.py uses opensimplex wrapper functions
  2. PointClusterer noise-based distribution works correctly
  3. Obstacle-aware clustering functional
  4. No regressions in point distribution patterns
**Plans**: 2 plans

Plans:
- [x] 72-01-PLAN.md — Verify noise import migration (already done in Phase 71)
- [x] 72-02-PLAN.md — Verify PointClusterer uses numpy.random (no migration needed)

---

### Phase 73: Entry Point Creation
**Goal**: Create clean generate_grass() function as primary Maya workflow entry point
**Depends on**: Phase 71, Phase 72
**Requirements**: REQ-004
**Success Criteria** (what must be TRUE):
  1. `from maya_grass import generate_grass` works in Maya script editor
  2. Function accepts terrain mesh name and grass geometry name as required params
  3. Sensible defaults for count (5000), wind_strength (2.5), scale variation (0.8-1.2)
  4. Returns MASH network name for further manipulation
  5. Module docstring documents usage workflow
**Plans**: 2 plans

Plans:
- [x] 73-01-PLAN.md — Create generate_grass() facade function with validation helpers
- [x] 73-02-PLAN.md — Add unit tests for validation and verify import chain

---

### Phase 74: Dependency Cleanup
**Goal**: Remove noise library dependency, update requirements for opensimplex
**Depends on**: Phase 71, Phase 72
**Requirements**: REQ-001 (completion)
**Success Criteria** (what must be TRUE):
  1. noise library removed from requirements.in
  2. opensimplex>=0.4.5 present in requirements.in
  3. No import errors when loading maya_grass module
  4. Module loads successfully in Maya 2023+ Python environment
**Plans**: 1 plan

Plans:
- [x] 74-01-PLAN.md — Remove noise from requirements and verify import chain

---

### Phase 75: Viewport Animation Verification
**Goal**: Confirm grass animates correctly in Maya viewport during playback
**Depends on**: Phase 73, Phase 74
**Requirements**: REQ-005
**Success Criteria** (what must be TRUE):
  1. Wind animation visible in viewport during timeline playback
  2. Animation responds to timeline scrubbing (frame changes update grass)
  3. Works in DG, Serial, and Parallel evaluation modes
  4. MASH Signal node properly connected to time1 node
**Plans**: 2 plans

Plans:
- [ ] 75-01: Create test scene with terrain and grass geometry
- [ ] 75-02: Test animation in all three evaluation modes (DG, Serial, Parallel)

---

### Phase 76: Render Animation Verification
**Goal**: Confirm grass animation renders correctly in playblast and batch render
**Depends on**: Phase 75
**Requirements**: REQ-006
**Success Criteria** (what must be TRUE):
  1. Playblast captures animated grass (not static)
  2. Batch render shows animation across frame sequence
  3. Envelope attribute animated if needed to force per-frame evaluation
  4. Frame 1, 12, 24 renders show distinct wind positions
**Plans**: 2 plans

Plans:
- [ ] 76-01: Verify playblast captures animation
- [ ] 76-02: Verify batch render frame sequence shows wind variation

---

### Phase 77: Integration Testing
**Goal**: Test complete end-to-end workflow from import to rendered output
**Depends on**: Phase 76
**Requirements**: REQ-004, REQ-005, REQ-006 (validation)
**Success Criteria** (what must be TRUE):
  1. Full workflow script executes without errors
  2. Workflow reproducible with different terrain/grass geometry
  3. Generated network can be modified after creation
  4. Undo works for grass generation operations
**Plans**: 2 plans

Plans:
- [ ] 77-01: Create integration test script with multiple terrain configurations
- [ ] 77-02: Test undo behavior and network modification

---

### Phase 78: Documentation and Polish
**Goal**: Usage documentation and edge case handling
**Depends on**: Phase 77
**Requirements**: REQ-004 (documentation)
**Success Criteria** (what must be TRUE):
  1. README documents installation steps for Maya
  2. Common error messages are helpful (missing terrain, no faces, etc.)
  3. Example scripts provided for typical use cases
  4. Known limitations documented (Maya version requirements, evaluation modes)
**Plans**: 2 plans

Plans:
- [ ] 78-01: Write installation and usage documentation
- [ ] 78-02: Add error handling for common edge cases

---

### Phase 79: Obstacle Proximity Density
**Goal**: Expose proximity_density_boost parameter in generate_grass() for foot traffic avoidance effect
**Depends on**: Phase 73
**Requirements**: REQ-007
**Success Criteria** (what must be TRUE):
  1. Grass density is higher within configurable radius of obstacles
  2. Density falloff from obstacle edge is smooth (not abrupt)
  3. Effect simulates foot traffic avoidance (people walk around objects, grass grows taller)
  4. Works with existing obstacle detection system
  5. Parameter exposed in generate_grass() function
**Plans**: 1 plan

Plans:
- [ ] 79-01-PLAN.md — Add proximity_density_boost parameter to generate_grass() with validation and tests

---

## Progress

**Execution Order:**
Phases execute in dependency order: 70 -> 71 -> 72 -> 73 -> 74 -> **79** -> 75 -> 76 -> 77 -> 78

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 70. Noise Wrapper Foundation | 2/2 | Complete | 2026-01-20 |
| 71. FlowField Migration | 2/3 | In Progress (gap closure) | - |
| 72. Flow Field Migration | 2/2 | Complete | 2026-01-20 |
| 73. Entry Point Creation | 2/2 | Complete | 2026-01-20 |
| 74. Dependency Cleanup | 1/1 | Complete | 2026-01-20 |
| 75. Viewport Animation Verification | 0/2 | Not started | - |
| 76. Render Animation Verification | 0/2 | Not started | - |
| 77. Integration Testing | 0/2 | Not started | - |
| 78. Documentation and Polish | 0/2 | Not started | - |
| 79. Obstacle Proximity Density | 0/1 | Not started | - |

---

## Requirement Traceability

| Requirement | Phase(s) | Description |
|-------------|----------|-------------|
| REQ-001 | 70, 74 | Replace noise library with opensimplex |
| REQ-002 | 71 | Update FlowField for opensimplex |
| REQ-003 | 72 | Update flow field noise usage |
| REQ-004 | 73, 77, 78 | Single entry point function |
| REQ-005 | 75, 77 | Verify viewport animation |
| REQ-006 | 76, 77 | Verify render animation |
| REQ-007 | 79 | Obstacle proximity density (foot traffic effect) |

---

*Roadmap created: 2026-01-20*
*Phase 70 planned: 2026-01-20*
*Phase 70 completed: 2026-01-20*
*Phase 71 planned: 2026-01-20*
*Phase 71 gap closure: 2026-01-22*
*Phase 72 planned: 2026-01-20*
*Phase 72 completed: 2026-01-20*
*Phase 73 planned: 2026-01-20*
*Phase 73 completed: 2026-01-20*
*Phase 74 planned: 2026-01-20*
*Phase 74 completed: 2026-01-20*
*Phase 79 planned: 2026-01-20*
