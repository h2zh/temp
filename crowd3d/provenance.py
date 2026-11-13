"""The provenance vocabulary: where a value of the clay model comes from (docs/STYLIZED_MODEL.md,
docs/STRUCTURE_FIRST.md §2.4.1). Every face of a published model carries one of these per PROVENANCE_KEYS, every
parameter of the structure one as its source.

Plain constants, no import (D34 part C: moved out of crowd3d/stylize.py unchanged, so that crowd3d.structure and the
judges need not import the module that imports them; crowd3d.stylize imports every name back).
"""

PROV_OSM = "osm-footprint"           # outline from OpenStreetMap (ODbL)
PROV_SFM = "sfm-points"              # height, outline or roof shape measured from the reconstruction's points
PROV_OSM_HEIGHT = "osm-height"       # OSM height=* or building:levels=* tag
PROV_WIKIDATA = "wikidata-height"    # Wikidata P2048 (height)
PROV_INFERRED = "inferred"           # not measured on this face: carried over from measured parts of its block
PROV_DEFAULT = "default-style"       # neither measured nor tagged: a neutral style default; every face's appearance
                                     # (D29: the clay look; `observed` says whether photos saw a wall)
# structure-first massing (crowd3d.structure, docs/STRUCTURE_FIRST.md §2.4.1): facts and fact-derived conventions
PROV_S3DB = "osm-s3db"               # outline / height / roof shape tagged on an OSM building:part (Simple 3D Buildings)
PROV_LEVELS = "osm-levels"           # building:levels / building:min_level / roof:levels x 3 m
PROV_ROOF_ANGLE = "osm-roof-angle"   # roof height from roof:angle and the part's half-width
PROV_SHAPE_DEFAULT = "s3db-shape-default"   # roof:shape tagged, roof height not: the shape's conventional proportion (D17)
PROV_SUPPORT = "osm-s3db-support"    # a main-outline gap's top: the min_height of the raised parts resting on it
PROV_GROUND = "osm-default-ground"   # no min_height tagged: the S3DB default 0 (on the ground)
PROV_CURATOR = "curator"             # entered or approved by a curator (same parameter format)
PROV_FITTED = "photo-fitted"         # changed by photo evidence (a curator-confirmed photo fit)
PROV_FITTED_AUTO = "photo-fitted (auto)"   # changed automatically: >= 2 photo channels agree (D15, on by default;
                                           # the map value and all evidence archived, reversible; crowd3d.judge)
PROV_CORRECTED_AUTO = "photo-corrected (auto)"   # D25 (strong photo evidence corrects OSM): the full photo value V
                                                 # (or a measured tilt); map value, evidence and the registration box
                                                 # archived, reversible (crowd3d.correct)
PROV_CORRECTED_LB = "photo-corrected (auto, lower bound)"   # D25: only the guaranteed lower bound (a height whose box
                                                            # lower bound sits below the midpoint, every D25-B block)
PROVENANCES = (PROV_OSM, PROV_SFM, PROV_OSM_HEIGHT, PROV_WIKIDATA, PROV_INFERRED, PROV_DEFAULT,
               PROV_S3DB, PROV_LEVELS, PROV_ROOF_ANGLE, PROV_SHAPE_DEFAULT, PROV_SUPPORT, PROV_GROUND, PROV_CURATOR,
               PROV_FITTED, PROV_FITTED_AUTO, PROV_CORRECTED_AUTO, PROV_CORRECTED_LB)
PROVENANCE_KEYS = ("outline", "height", "appearance")   # every face carries all three
