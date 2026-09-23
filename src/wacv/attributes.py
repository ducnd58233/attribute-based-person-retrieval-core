"""The 40 UPAR attributes, in the exact order the challenge uses.

Single source of truth: every gt.csv column, every query vector index, and
every model output index follows this order. See the parent workspace's
PROBLEM-DETAILS.md, section 5, for the authoritative list.
"""

ATTRIBUTE_NAMES: tuple[str, ...] = (
    "Age-Young",
    "Age-Adult",
    "Age-Old",
    "Gender-Female",
    "Hair-Length-Short",
    "Hair-Length-Long",
    "Hair-Length-Bald",
    "UpperBody-Length-Short",
    "UpperBody-Color-Black",
    "UpperBody-Color-Blue",
    "UpperBody-Color-Brown",
    "UpperBody-Color-Green",
    "UpperBody-Color-Grey",
    "UpperBody-Color-Orange",
    "UpperBody-Color-Pink",
    "UpperBody-Color-Purple",
    "UpperBody-Color-Red",
    "UpperBody-Color-White",
    "UpperBody-Color-Yellow",
    "UpperBody-Color-Other",
    "LowerBody-Length-Short",
    "LowerBody-Color-Black",
    "LowerBody-Color-Blue",
    "LowerBody-Color-Brown",
    "LowerBody-Color-Green",
    "LowerBody-Color-Grey",
    "LowerBody-Color-Orange",
    "LowerBody-Color-Pink",
    "LowerBody-Color-Purple",
    "LowerBody-Color-Red",
    "LowerBody-Color-White",
    "LowerBody-Color-Yellow",
    "LowerBody-Color-Other",
    "LowerBody-Type-Trousers&Shorts",
    "LowerBody-Type-Skirt&Dress",
    "Accessory-Backpack",
    "Accessory-Bag",
    "Accessory-Glasses-Normal",
    "Accessory-Glasses-Sun",
    "Accessory-Hat",
)

NUM_ATTRIBUTES = len(ATTRIBUTE_NAMES)
