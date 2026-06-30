import matplotlib.pyplot as plt

from ingkit import myplot


def test_pyplot_remains_available_from_myplot():
    assert myplot.pyplot is plt


def test_style_wrapper_functions():
    myplot.use_style("default")
    myplot.use_my_default()

    assert "my_default" in myplot.styles.available_styles(user=True)

    myplot.use_style("default")


def test_my_default2_color_cycle():
    params = myplot.styles.get_rcParams("my_default2")
    colors = params["axes.prop_cycle"].by_key()["color"]

    assert colors == ["magenta", "cyan", "lime", "orange", "purple"]
