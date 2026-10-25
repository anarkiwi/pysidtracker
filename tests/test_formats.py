def test_sidformat_variant_info_and_sites_hooks():
    from pysidtracker import SidFormat
    from pysidtracker.facets import DEFAULT_SOURCE, Entry, Variant

    var = Variant({"driver": Entry(3, DEFAULT_SOURCE)})
    fmt = SidFormat(
        "x", None, None, variant=lambda m: var, facet_sites=lambda m: {0x1000: "driver"}
    )
    assert fmt.variant_info(None) == {"driver": 3, "driver.src": "default"}
    assert fmt.sites(None) == {0x1000: "driver"}
    bare = SidFormat("y", None, None)
    assert bare.variant_info(None) == {} and bare.sites(None) == {}
