import distutils.util


def print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))


def add_arguments(argname, type, default, help, argparser, shortname=None, **kwargs):
    type = distutils.util.strtobool if type == bool else type
    args = ["--" + argname]
    if shortname:
        args.insert(0, shortname)
    argparser.add_argument(*args,
                           default=default,
                           type=type,
                           help=help + ' Default: %(default)s.',
                           **kwargs)
