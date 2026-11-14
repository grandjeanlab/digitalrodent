"""
Single source of truth for the dual-regression component labels.

New ICA labels (Zerbi). The parcellation list is 0-based, the DR components
are 1-based:  comp{n} = parcellation index n-1  (comp1 = Pir, comp2 = M1, ...).

Inside the normative model the response variables stay "comp1".."comp18"
(the saved model, z-scores and statistics files use those names); these
labels are only used for figures and tables.

Import from any .qmd in Analysis/ with:
    from component_labels import (COMPONENTS, ACRONYM, NAME,
                                   SELECTED_COMPONENTS, COMPONENT_COLORS)
"""

# (acronym, full name), index 0..17  ->  comp1..comp18
PARCELLATION = [
    ("Pir", "Piriform cortex"),                                              # comp1
    ("M1", "Primary motor cortex"),                                          # comp2
    ("S1uLp + S2", "S1 upper lip + secondary somatosensory cortex"),         # comp3
    ("S1 HL + FL", "S1 hindlimb + forelimb"),                                # comp4
    ("S1 FL + S1 BF", "S1 forelimb + barrel field"),                         # comp5
    ("S1 BF", "S1 barrel field"),                                            # comp6
    ("V2 + RSD", "Secondary visual + dorsal retrosplenial cortex"),          # comp7
    ("AUd + AUv/TeA-ECT", "Auditory + temporal association/ectorhinal cortex"),  # comp8
    ("Cg2", "Cingulate cortex, area 2"),                                     # comp9
    ("Cg1 + RS", "Cingulate cortex, area 1 + retrosplenial cortex"),         # comp10
    ("Dorsal HP/SC", "Dorsal hippocampus / superior colliculus"),            # comp11
    ("Ventral HP", "Ventral hippocampus"),                                   # comp12
    ("CPu", "Caudate putamen"),                                              # comp13
    ("CPu + En", "Caudate putamen + endopiriform nucleus"),                  # comp14
    ("Septum + VP", "Septum + ventral pallidum"),                             # comp15
    ("Amygdala", "Amygdala"),                                                # comp16
    ("Thalamus", "Thalamus"),                                                # comp17
    ("MO + GI", "Medial orbital + granular insular cortex"),                 # comp18
]

COMPONENTS = [f"comp{i + 1}" for i in range(len(PARCELLATION))]
ACRONYM = {c: acr for c, (acr, _) in zip(COMPONENTS, PARCELLATION)}
NAME = {c: name for c, (_, name) in zip(COMPONENTS, PARCELLATION)}

# Networks shown in the main figures.
# NOTE: with the old labels comp11 was called "Ventral HP"; with the new
# Zerbi labels comp11 = dorsal HP / superior colliculus and the ventral
# hippocampus is comp12.
SELECTED_COMPONENTS = ["comp2", "comp3", "comp7", "comp10", "comp12"]

COMPONENT_COLORS = {
    "comp2": "#8E6C8A",   # M1, motor
    "comp3": "#4682B4",   # S1uLp + S2, somatosensory
    "comp7": "#C9A227",   # V2 + RSD, visual
    "comp10": "#F4A460",  # Cg1 + RS
    "comp12": "#6D9F8C",  # Ventral HP
    "comp11": "#9C6B4E",  # Dorsal HP / SC (if added to the selection)
}
