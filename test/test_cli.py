from ingkit.utils.cli import CLI


def test_cli_wraps_argument_parser():
    cli = CLI(description="demo")
    action = cli.add_argument("--count", type=int, default=1)

    args = cli.parse_args(["--count", "3"])

    assert action.dest == "count"
    assert args.count == 3
