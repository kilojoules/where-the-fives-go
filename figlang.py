"""One vocabulary for every figure in this repo.

Three quantities are measured across all the figures here, and they had drifted into seven
names -- "information", "leak", "readout" were the SAME measurement, and "LEAK" appeared on
one axis meaning raw AUC and on another meaning distance from chance. A reader comparing two
plots could not tell whether an axis change was a change of quantity or a change of wording.

The three quantities, and nothing else:

  READOUT      AUC over the model's own raw logit 5, module DETACHED. What ships, and what a
               user who only sees outputs can rank by. Chance is 0.50; below 0.50 is an
               anti-ranking, which is still information with the sign flipped, so where the
               figure wants "how much is left" it shows DISTANCE FROM CHANCE, |AUC - 0.50|.

  CAPABILITY   recall on 5s with the module ATTACHED. What the method preserved for
               legitimate use. A different mask from READOUT, not a rescaling of it.

  PROBE        AUC of a binary 5-vs-rest linear probe fitted on all ten logits, module
               DETACHED. What an attacker who can fit a readout gets. Bounded below by
               READOUT, since "use logit 5" is one of the combinations available to it.

Never "leak" or "information" as an axis name: both were aliases for READOUT and reading two
figures side by side suggested three measurements where there were two.
"""

# axis labels -- raw scale
READOUT = "READOUT — AUC over the model's own logit 5  (module detached)"
CAPABILITY = "CAPABILITY — recall on 5s  (module attached)"
PROBE = "PROBE — AUC of a linear probe on all ten logits  (module detached)"
GAP = "GAP — how much the probe beats the readout  (probe − readout)"

# axis labels -- folded at chance, for "how much is left"
READOUT_FOLDED = ("READOUT above chance  |AUC(logit 5) − 0.50|"
                  "        ←  less left behind")
PROBE_FOLDED = "PROBE above chance  |AUC(probe) − 0.50|        ←  less left behind"
CAPABILITY_DIR = CAPABILITY + "        better  →"

# the sentence that has to appear wherever a folded axis does
FOLD_NOTE = ("Folded at chance because below-chance is still information with the sign "
             "flipped: an anti-ranking is recoverable by inverting it.")

# the sentence that has to appear wherever both READOUT and PROBE appear
PROBE_NOTE = ("The probe can never score below the readout: ‘use logit 5 alone’ is one "
              "of the combinations available to it.")

# plain-language gloss for the Pareto figures, replacing "non-dominated"
PARETO_NOTE = ("Only the best trade-offs are shown — a setting appears if nothing beat it on "
               "BOTH axes at once.")

# house colours, so arms are the same colour in every figure
C = {"dense": "#2a78d6", "filtered": "#eb6834", "gram": "#1baf7a",
     "coreadapt": "#6250d6", "grey": "#9a9a92", "ink": "#333333"}
