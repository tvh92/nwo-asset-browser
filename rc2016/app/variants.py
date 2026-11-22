"""Model variants the browser can hide: alternate versions of a model listed next to it, test or legacy folders, and
models with nothing to draw.

A suffix variant is <folder>/<name>_<suffix>.model whose base <folder>/<name>.model is also listed, so a model that
only exists with the suffix stays listed. The kinds come from the Rift Apart file names; the R&C 2016 archive
has none of these suffixes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariantKind:
    key: str
    label: str
    hidden_by_default: bool
    suffixes: tuple[str, ...] = ()   # "_topiary" matches <name>_topiary.model when <name>.model is listed
    folders: tuple[str, ...] = ()    # path prefixes, for whole folders
    flag: str = ""                   # a record attribute that marks the kind (PipelineModelCandidate.no_geometry)
    words: tuple[str, ...] = ()      # search words that show the kind again


VARIANT_KINDS = (
    VariantKind("topiary", "Topiary versions (_topiary)", True, suffixes=("_topiary",)),
    VariantKind("test", "Test folder (characters/test)", True, folders=("characters/test/",)),
    VariantKind("legacy", "Legacy folder (older-game models)", False, folders=("legacy/",)),
    VariantKind("skeleton", "Skeleton-only models (_skeleton)", False, suffixes=("_skeleton",)),
    VariantKind("havok", "Havok physics versions (_havok)", False, suffixes=("_havok",)),
    VariantKind("duplicate", "Static and animated copies (_anim, _static, _nonstatic)", False,
                suffixes=("_anim", "_static", "_nonstatic")),
    VariantKind("snow", "Snow versions (_snow)", False, suffixes=("_snow",)),
    VariantKind("economy", "Economy versions (_economy)", False, suffixes=("_economy",)),
    VariantKind("special", "Cutscene, hologram and dimension-swap versions", False,
                suffixes=("_cine", "_hologram", "_dimensionswap")),
    VariantKind("no_geometry", "Models without geometry (collision, joints or placeholders)", True, flag="no_geometry",
                words=("geometry", "empty")),
)


def variant_kind(path: str, listed: set[str] | frozenset[str], record=None) -> str | None:
    """The key of the variant kind of path, or None. listed holds every listed model path, casefolded; a kind with a flag
    matches a record whose attribute of that name is true, before any path rule."""
    for kind in VARIANT_KINDS:
        if kind.flag and getattr(record, kind.flag, False):
            return kind.key
    path = path.replace("\\", "/").casefold()
    if path.endswith(".model"):   # a suffix variant first, so a legacy topiary copy is topiary
        stem = path[:-len(".model")]
        for kind in VARIANT_KINDS:
            for suffix in kind.suffixes:
                if stem.endswith(suffix) and stem[:-len(suffix)] + ".model" in listed:
                    return kind.key
    for kind in VARIANT_KINDS:
        if kind.folders and path.startswith(kind.folders):
            return kind.key
    return None


def search_reveals(kind_key: str, terms: list[str]) -> bool:
    """A hidden kind shows again while a search word names it ('topiary', 'snow', 'legacy', ...)."""
    kind = next(k for k in VARIANT_KINDS if k.key == kind_key)
    names = ({s.lstrip("_") for s in kind.suffixes} | {f.rstrip("/").rpartition("/")[2] for f in kind.folders}
             | set(kind.words))
    return any(term.lstrip("_") in names for term in terms)
